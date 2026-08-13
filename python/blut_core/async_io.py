"""Bounded, backpressured asynchronous I/O for cookbook runtimes.

Bounded workers are daemons so an exceptional interpreter shutdown cannot hang
on an idle worker. Successful callers must still call :meth:`AsyncSink.close`;
that is the only boundary that drains accepted work and surfaces worker errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from queue import Queue
from threading import BoundedSemaphore, Condition, Thread, local
from typing import Callable, Generic, TypeVar, cast


__all__ = [
    "AsyncSink",
    "AsyncSinkWorkerError",
    "Bounded",
    "Inline",
    "ItemTooLargeError",
    "SinkClosedError",
]


ItemT = TypeVar("ItemT")


def _identity(item: ItemT) -> ItemT:
    return item


class ItemTooLargeError(ValueError):
    """A submitted item exceeds the sink's declared retention limit."""

    def __init__(self, size_bytes: int, max_item_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_item_bytes = max_item_bytes
        super().__init__(f"item is {size_bytes} bytes; limit is {max_item_bytes}")


class SinkClosedError(RuntimeError):
    """Work was submitted after sink shutdown began."""


class AsyncSinkWorkerError(RuntimeError):
    """A sink worker failed while delivering an accepted item."""

    def __init__(self, worker_error_type: str, message: str) -> None:
        self.worker_error_type = worker_error_type
        super().__init__(message)


@dataclass(frozen=True)
class _WorkerFailure:
    worker_error_type: str
    message: str

    @classmethod
    def capture(cls, error: BaseException) -> _WorkerFailure:
        error_type = type(error)
        worker_error_type = f"{error_type.__module__}.{error_type.__qualname__}"
        try:
            message = str(error)
        except BaseException:
            message = "worker failure with an unprintable message"
        return cls(worker_error_type=worker_error_type, message=message)

    def to_exception(self) -> AsyncSinkWorkerError:
        return AsyncSinkWorkerError(self.worker_error_type, self.message)


@dataclass(frozen=True)
class Inline:
    """Run the installed handler synchronously on the submitting thread."""


@dataclass(frozen=True)
class Bounded:
    """Run work on one worker with bounded retained residency."""

    capacity: int
    max_item_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("capacity must be an integer")
        if isinstance(self.max_item_bytes, bool) or not isinstance(
            self.max_item_bytes, int
        ):
            raise TypeError("max_item_bytes must be an integer")
        if self.capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if self.max_item_bytes <= 0:
            raise ValueError("max_item_bytes must be greater than zero")


_STOP = object()


class AsyncSink(Generic[ItemT]):
    """Deliver items to one handler under an explicit, close-required mode."""

    def __init__(
        self,
        handler: Callable[[ItemT], None],
        mode: Inline | Bounded,
        *,
        item_size: Callable[[ItemT], int] | None = None,
        retain: Callable[[ItemT], ItemT] = _identity,
        retained_size: Callable[[ItemT], int] | None = None,
        thread_name: str = "blut-async-io",
    ) -> None:
        self._handler = handler
        self._mode = mode
        self._item_size = item_size
        self._retain = retain
        self._retained_size = retained_size
        self._closed = False
        self._failure: _WorkerFailure | None = None
        self._state = Condition()
        self._active_submits = 0
        self._stop_enqueued = False
        self._callback_state = local()
        self._queue: Queue[ItemT | object] | None = None
        self._slots: BoundedSemaphore | None = None
        self._worker: Thread | None = None
        if isinstance(mode, Bounded):
            self._queue = Queue(maxsize=mode.capacity)
            self._slots = BoundedSemaphore(mode.capacity)
            self._worker = Thread(target=self._run, name=thread_name, daemon=True)
            self._worker.start()

    def submit(self, item: ItemT, *, size_bytes: int | None = None) -> None:
        """Deliver one item according to the configured mode."""

        if getattr(self._callback_state, "active", False):
            raise RuntimeError("submit cannot be called from a sink callback")

        with self._state:
            self._raise_if_failed_locked()
            if self._closed:
                raise SinkClosedError("sink is closed")
            self._active_submits += 1

        acquired_slot = False
        try:
            if isinstance(self._mode, Inline):
                self._invoke_handler(self._invoke_callback(self._retain, item))
                return

            if size_bytes is None and self._item_size is not None:
                size_bytes = self._invoke_callback(self._item_size, item)
            if size_bytes is None and self._retained_size is None:
                raise TypeError(
                    "bounded submit requires size_bytes, item_size, or retained_size"
                )
            if size_bytes is not None:
                if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
                    raise TypeError("item size must be an integer byte count")
                if size_bytes < 0:
                    raise ValueError("item size must not be negative")
                if size_bytes > self._mode.max_item_bytes:
                    raise ItemTooLargeError(size_bytes, self._mode.max_item_bytes)

            slots = cast(BoundedSemaphore, self._slots)
            queue = cast(Queue[ItemT | object], self._queue)
            slots.acquire()
            acquired_slot = True
            with self._state:
                self._raise_if_failed_locked()
                if self._closed:
                    raise SinkClosedError("sink is closed")
            retained = self._invoke_callback(self._retain, item)
            if self._retained_size is not None:
                retained_bytes = self._invoke_callback(self._retained_size, retained)
                if isinstance(retained_bytes, bool) or not isinstance(retained_bytes, int):
                    raise TypeError("retained item size must be an integer byte count")
                if retained_bytes < 0:
                    raise ValueError("retained item size must not be negative")
                if retained_bytes > self._mode.max_item_bytes:
                    raise ItemTooLargeError(retained_bytes, self._mode.max_item_bytes)
            queue.put(retained)
            acquired_slot = False
        finally:
            if acquired_slot:
                cast(BoundedSemaphore, self._slots).release()
            with self._state:
                self._active_submits -= 1
                self._state.notify_all()

    def close(self) -> None:
        """Drain accepted work and stop the sink."""

        if getattr(self._callback_state, "active", False):
            raise RuntimeError("close cannot be called from a sink callback")

        with self._state:
            self._closed = True
            while self._active_submits:
                self._state.wait()
            if self._worker is not None and not self._stop_enqueued:
                cast(Queue[ItemT | object], self._queue).put(_STOP)
                self._stop_enqueued = True

        if self._worker is not None:
            self._worker.join()
        self._raise_if_failed()

    def _run(self) -> None:
        queue = cast(Queue[ItemT | object], self._queue)
        slots = cast(BoundedSemaphore, self._slots)
        while True:
            item = queue.get()
            if item is _STOP:
                return
            try:
                self._invoke_handler(cast(ItemT, item))
            except BaseException as error:
                with self._state:
                    if self._failure is None:
                        self._failure = _WorkerFailure.capture(error)
            finally:
                del item
                slots.release()

    def _invoke_handler(self, item: ItemT) -> None:
        self._invoke_callback(self._handler, item)

    def _invoke_callback(self, callback, item):
        previous = getattr(self._callback_state, "active", False)
        self._callback_state.active = True
        try:
            return callback(item)
        finally:
            self._callback_state.active = previous

    def _raise_if_failed_locked(self) -> None:
        if self._failure is not None:
            raise self._failure.to_exception()

    def _raise_if_failed(self) -> None:
        with self._state:
            failure = self._failure
        if failure is not None:
            raise failure.to_exception()
