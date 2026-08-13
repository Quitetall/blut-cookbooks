"""Public contract tests for bounded cookbook async I/O."""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
import time
import weakref

import pytest

from blut_core.async_io import (
    AsyncSink,
    Bounded,
    Inline,
    ItemTooLargeError,
    SinkClosedError,
)
from blut_core.metric_log import MetricLog


@pytest.mark.parametrize(
    ("capacity", "max_item_bytes", "error"),
    [
        (0, 4, ValueError),
        (1, 0, ValueError),
        (True, 4, TypeError),
        (1, 4.5, TypeError),
    ],
)
def test_bounded_mode_requires_positive_integer_limits(
    capacity: object,
    max_item_bytes: object,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        Bounded(capacity=capacity, max_item_bytes=max_item_bytes)  # type: ignore[arg-type]


def test_inline_mode_handles_items_on_the_calling_thread() -> None:
    caller = threading.get_ident()
    handled: list[tuple[str, int]] = []

    sink = AsyncSink(lambda item: handled.append((item, threading.get_ident())), Inline())

    sink.submit("metric")
    assert handled == [("metric", caller)]
    sink.close()


def test_bounded_mode_delivers_every_item_fifo_on_one_worker() -> None:
    caller = threading.get_ident()
    handled: list[tuple[int, int]] = []
    sink = AsyncSink(
        lambda item: handled.append((item, threading.get_ident())),
        Bounded(capacity=3, max_item_bytes=16),
    )

    for item in range(6):
        sink.submit(item, size_bytes=8)
    sink.close()

    assert [item for item, _thread in handled] == list(range(6))
    assert {thread for _item, thread in handled} != {caller}


def test_capacity_counts_running_work_and_is_acquired_before_retaining() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_measured = threading.Event()
    second_retained = threading.Event()

    def handle(item: str) -> None:
        if item == "first":
            first_started.set()
            assert release_first.wait(timeout=1)

    def item_size(item: str) -> int:
        if item == "second":
            second_measured.set()
        return len(item)

    def retain(item: str) -> str:
        if item == "second":
            second_retained.set()
        return item

    sink = AsyncSink(
        handle,
        Bounded(capacity=1, max_item_bytes=16),
        item_size=item_size,
        retain=retain,
    )
    sink.submit("first")
    assert first_started.wait(timeout=1)

    second_submit = threading.Thread(target=lambda: sink.submit("second"))
    second_submit.start()
    assert second_measured.wait(timeout=1)
    assert not second_retained.wait(timeout=0.05)

    release_first.set()
    second_submit.join(timeout=1)
    assert not second_submit.is_alive()
    assert second_retained.is_set()
    sink.close()


def test_oversized_item_is_rejected_before_it_is_retained() -> None:
    retained: list[bytes] = []
    sink = AsyncSink(
        lambda _item: None,
        Bounded(capacity=1, max_item_bytes=4),
        retain=lambda item: retained.append(item) or item,
    )

    with pytest.raises(ItemTooLargeError, match=r"5 bytes; limit is 4"):
        sink.submit(b"12345", size_bytes=5)

    assert retained == []
    sink.close()


def test_exact_retained_size_is_checked_after_admitted_snapshot() -> None:
    retained: list[bytes] = []
    handled: list[bytes] = []

    def snapshot(item: str) -> bytes:
        payload = item.encode("utf-8")
        retained.append(payload)
        return payload

    sink = AsyncSink(
        handled.append,
        Bounded(capacity=1, max_item_bytes=4),
        retain=snapshot,
        retained_size=len,
    )

    with pytest.raises(ItemTooLargeError, match=r"5 bytes; limit is 4"):
        sink.submit("12345")

    sink.close()
    assert retained == [b"12345"]
    assert handled == []


def test_close_surfaces_the_first_worker_failure_after_drain() -> None:
    handled: list[str] = []

    def handle(item: str) -> None:
        handled.append(item)
        if item == "bad":
            raise RuntimeError("disk full")

    sink = AsyncSink(handle, Bounded(capacity=3, max_item_bytes=16))
    sink.submit("bad", size_bytes=3)
    sink.submit("after", size_bytes=5)

    with pytest.raises(RuntimeError, match="disk full"):
        sink.close()
    with pytest.raises(RuntimeError, match="disk full"):
        sink.close()

    assert handled == ["bad", "after"]


def test_completed_item_is_released_before_capacity_is_returned() -> None:
    class Payload:
        pass

    handled = threading.Event()
    sink = AsyncSink(
        lambda _item: handled.set(),
        Bounded(capacity=1, max_item_bytes=1),
    )
    payload = Payload()
    witness = weakref.ref(payload)

    sink.submit(payload, size_bytes=1)
    assert handled.wait(timeout=1)
    time.sleep(0.02)
    del payload
    gc.collect()

    assert witness() is None
    sink.close()


def test_worker_failure_does_not_retain_payload_or_accumulate_tracebacks() -> None:
    class Payload:
        pass

    def fail(_item: Payload) -> None:
        raise RuntimeError("disk full")

    sink = AsyncSink(fail, Bounded(capacity=1, max_item_bytes=1))
    payload = Payload()
    witness = weakref.ref(payload)
    sink.submit(payload, size_bytes=1)

    traceback_depths = []
    for _ in range(4):
        try:
            sink.close()
        except RuntimeError as error:
            depth = 0
            traceback = error.__traceback__
            while traceback is not None:
                depth += 1
                traceback = traceback.tb_next
            traceback_depths.append(depth)

    del payload
    gc.collect()

    assert witness() is None
    assert len(set(traceback_depths)) == 1


def test_blocked_submit_surfaces_worker_failure_before_retaining() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_measured = threading.Event()
    retained: list[str] = []
    submit_errors: list[BaseException] = []

    def handle(item: str) -> None:
        if item == "first":
            first_started.set()
            assert release_first.wait(timeout=1)
            raise RuntimeError("first write failed")

    def item_size(item: str) -> int:
        if item == "second":
            second_measured.set()
        return len(item)

    sink = AsyncSink(
        handle,
        Bounded(capacity=1, max_item_bytes=16),
        item_size=item_size,
        retain=lambda item: retained.append(item) or item,
    )
    sink.submit("first")
    assert first_started.wait(timeout=1)

    def submit_second() -> None:
        try:
            sink.submit("second")
        except BaseException as error:
            submit_errors.append(error)

    producer = threading.Thread(target=submit_second)
    producer.start()
    assert second_measured.wait(timeout=1)
    release_first.set()
    producer.join(timeout=1)

    assert not producer.is_alive()
    assert [str(error) for error in submit_errors] == ["first write failed"]
    assert retained == ["first"]
    with pytest.raises(RuntimeError, match="first write failed"):
        sink.close()


def test_close_rejects_a_blocked_submit_before_enqueuing_stop() -> None:
    first_started = threading.Event()
    release_first = threading.Event()
    second_measured = threading.Event()
    retained: list[str] = []
    handled: list[str] = []
    submit_errors: list[BaseException] = []

    def handle(item: str) -> None:
        handled.append(item)
        if item == "first":
            first_started.set()
            assert release_first.wait(timeout=1)

    def item_size(item: str) -> int:
        if item == "second":
            second_measured.set()
        return len(item)

    sink = AsyncSink(
        handle,
        Bounded(capacity=1, max_item_bytes=16),
        item_size=item_size,
        retain=lambda item: retained.append(item) or item,
    )
    sink.submit("first")
    assert first_started.wait(timeout=1)

    def submit_second() -> None:
        try:
            sink.submit("second")
        except BaseException as error:
            submit_errors.append(error)

    producer = threading.Thread(target=submit_second)
    producer.start()
    assert second_measured.wait(timeout=1)

    closer = threading.Thread(target=sink.close)
    closer.start()
    while True:
        try:
            sink.submit("oversized probe", size_bytes=17)
        except ItemTooLargeError:
            continue
        except SinkClosedError:
            break

    release_first.set()
    producer.join(timeout=1)
    closer.join(timeout=1)

    assert not producer.is_alive()
    assert not closer.is_alive()
    assert len(submit_errors) == 1
    assert isinstance(submit_errors[0], SinkClosedError)
    assert retained == ["first"]
    assert handled == ["first"]


def test_close_is_idempotent_and_late_submit_is_rejected() -> None:
    handled: list[str] = []
    sink = AsyncSink(handled.append, Bounded(capacity=1, max_item_bytes=16))
    sink.submit("done", size_bytes=4)

    sink.close()
    sink.close()

    assert handled == ["done"]
    with pytest.raises(SinkClosedError, match="sink is closed"):
        sink.submit("late", size_bytes=4)


def test_handler_cannot_close_its_own_sink_worker() -> None:
    sink: AsyncSink[str]

    def handle(_item: str) -> None:
        sink.close()

    sink = AsyncSink(handle, Bounded(capacity=1, max_item_bytes=16))
    sink.submit("reentrant", size_bytes=9)

    with pytest.raises(RuntimeError, match="close cannot be called from a sink callback"):
        sink.close()


@pytest.mark.parametrize(
    "script",
    [
        """
from blut_core.async_io import AsyncSink, Bounded

sink = None
def item_size(_item):
    try:
        sink.close()
    except RuntimeError:
        print("rejected")
    return 1

sink = AsyncSink(lambda _item: None, Bounded(1, 1), item_size=item_size)
sink.submit("x")
sink.close()
""",
        """
from blut_core.async_io import AsyncSink, Bounded

sink = None
def retain(item):
    try:
        sink.close()
    except RuntimeError:
        print("rejected")
    return item

sink = AsyncSink(lambda _item: None, Bounded(1, 1), retain=retain)
sink.submit("x", size_bytes=1)
sink.close()
""",
        """
from blut_core.async_io import AsyncSink, Bounded
import threading

sink = None
outer_started = threading.Event()
allow_recursive = threading.Event()
def handle(item):
    if item == "outer":
        outer_started.set()
        assert allow_recursive.wait(timeout=1)
        try:
            sink.submit("inner", size_bytes=1)
        except RuntimeError:
            print("rejected")

sink = AsyncSink(handle, Bounded(1, 5))
sink.submit("outer", size_bytes=5)
assert outer_started.wait(timeout=1)
allow_recursive.set()
sink.close()
""",
    ],
)
def test_same_sink_callback_reentrancy_is_rejected_without_deadlock(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )

    assert completed.stdout.strip() == "rejected"


def test_unclosed_sink_does_not_pin_interpreter_shutdown() -> None:
    script = """
from blut_core.async_io import AsyncSink, Bounded

AsyncSink(lambda _item: None, Bounded(capacity=1, max_item_bytes=1))
"""

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=2,
    )


def test_metric_log_keeps_legacy_best_effort_default(tmp_path, monkeypatch) -> None:
    metric_log = MetricLog("legacy", tmp_path)
    monkeypatch.setattr(
        metric_log,
        "_flush",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )

    metric_log.append({"epoch": 1})
    metric_log.close()


def test_metric_log_strict_mode_surfaces_append_and_close_failures(
    tmp_path, monkeypatch
) -> None:
    metric_log = MetricLog("authoritative", tmp_path, raise_on_error=True)
    monkeypatch.setattr(
        metric_log,
        "_flush",
        lambda: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        metric_log.append({"epoch": 1})
    with pytest.raises(OSError, match="disk full"):
        metric_log.close()
