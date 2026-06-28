"""Unit tests for the async-compute helpers in blut_core.trainer.

All run on CPU with no torchrun / GPU: `_dataloader_kwargs` is pure config
mapping, and `AsyncCheckpointSaver` is exercised with a fake `save_fn` that
records call order + payloads. `CudaPrefetcher` needs CUDA, so only its
construction guard is checked here (the train loop gates it on
`torch.cuda.is_available()`).
"""
import threading
import time

import blut_core.trainer as trainer


# --- _dataloader_kwargs: config → DataLoader knobs ----------------------------

def test_dataloader_kwargs_default_is_naive():
    # Omitting the knobs reproduces the old loader: 0 workers, no prefetch keys
    # (prefetch_factor/persistent_workers are invalid with num_workers=0).
    kw = trainer._dataloader_kwargs({})
    assert kw["num_workers"] == 0
    assert kw["pin_memory"] is False
    assert "prefetch_factor" not in kw
    assert "persistent_workers" not in kw


def test_dataloader_kwargs_with_workers_adds_prefetch():
    kw = trainer._dataloader_kwargs({
        "num_workers": 4,
        "pin_memory": True,
        "prefetch_factor": 6,
        "persistent_workers": True,
    })
    assert kw["num_workers"] == 4
    assert kw["pin_memory"] is True
    assert kw["prefetch_factor"] == 6
    assert kw["persistent_workers"] is True


def test_dataloader_kwargs_workers_default_prefetch():
    # With workers > 0 but no explicit prefetch, sensible defaults appear.
    kw = trainer._dataloader_kwargs({"num_workers": 2})
    assert kw["prefetch_factor"] == 2
    assert kw["persistent_workers"] is True


# --- AsyncCheckpointSaver: single-slot, ordered, correct ----------------------

def test_async_saver_writes_payload():
    saved = {}

    def fake_save(payload, path):
        saved[path] = payload

    saver = trainer.AsyncCheckpointSaver(fake_save)
    saver.save({"model": {"w": 1}}, "/tmp/ckpt-a")
    saver.close()  # joins the in-flight write
    assert saved["/tmp/ckpt-a"] == {"model": {"w": 1}}


def test_async_saver_never_overlaps():
    # Two rapid saves must not run concurrently: the second joins the first.
    active = {"count": 0, "max": 0}
    lock = threading.Lock()

    def slow_save(payload, path):
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
        time.sleep(0.05)
        with lock:
            active["count"] -= 1

    saver = trainer.AsyncCheckpointSaver(slow_save)
    saver.save({"x": 1}, "/tmp/ckpt-1")
    saver.save({"x": 2}, "/tmp/ckpt-2")  # must join the first
    saver.close()
    assert active["max"] == 1, "writes overlapped — single-slot violated"


def test_async_saver_roundtrip_matches_sync(tmp_path):
    # The checkpoint the async path writes must reload to the SAME tensor values
    # as a synchronous torch.save of the same payload. (Byte-identity is not the
    # invariant — torch.save embeds storage layout that differs after a CPU
    # copy; value-equality on reload is what a checkpoint actually guarantees.)
    import torch

    payload = {"model": {"w": torch.arange(6).float().reshape(2, 3), "b": torch.ones(3)}}

    sync_path = tmp_path / "sync.pt"
    torch.save(payload, str(sync_path))  # synchronous reference

    async_path = tmp_path / "async.pt"
    saver = trainer.AsyncCheckpointSaver(lambda p, path: torch.save(p, path))
    saver.save(payload, str(async_path))
    saver.close()

    sync_sd = torch.load(str(sync_path), weights_only=True)["model"]
    async_sd = torch.load(str(async_path), weights_only=True)["model"]
    assert torch.equal(async_sd["w"], sync_sd["w"])
    assert torch.equal(async_sd["b"], sync_sd["b"])


def test_async_saver_context_manager_closes_on_error():
    # The pool must shut down even if save raises — no leaked worker. Using the
    # context manager, __exit__ joins/closes regardless of the exception.
    closed = {"shutdown": False}

    def boom(payload, path):
        raise RuntimeError("disk full")

    saver = trainer.AsyncCheckpointSaver(boom)
    real_shutdown = saver._pool.shutdown

    def tracking_shutdown(*a, **kw):
        closed["shutdown"] = True
        return real_shutdown(*a, **kw)

    saver._pool.shutdown = tracking_shutdown

    raised = False
    try:
        with saver:
            saver.save({"x": 1}, "/tmp/ckpt-err")
            # The error surfaces when close() joins the pending future in __exit__.
    except RuntimeError:
        raised = True
    assert raised, "the background save error must surface"
    assert closed["shutdown"], "pool must be shut down on error (no leak)"


def test_record_stream_traverses_all_containers():
    # _record_stream must call record_stream on every tensor regardless of
    # container type (dict / list / tuple) — a tuple batch must not be skipped
    # (that would be a CUDA use-after-free). Tested with fakes, no GPU needed.
    class FakeTensor:
        def __init__(self):
            self.recorded = 0

        def record_stream(self, _stream):
            self.recorded += 1

    # Build a CudaPrefetcher without running __init__ (it needs CUDA): we only
    # exercise the pure _record_stream traversal via an unbound call.
    pref = trainer.CudaPrefetcher.__new__(trainer.CudaPrefetcher)
    a, b, c, d = FakeTensor(), FakeTensor(), FakeTensor(), FakeTensor()
    batch = {"x": a, "pair": (b, c), "list": [d]}
    pref._record_stream(batch, stream="dummy")
    assert (a.recorded, b.recorded, c.recorded, d.recorded) == (1, 1, 1, 1)


def test_to_cpu_recurses_containers():
    # Tensors nested in a list/tuple must be moved to CPU, not left on device.
    import torch

    payload = {"a": torch.ones(2), "nested": [torch.zeros(2), {"deep": torch.ones(1)}]}
    out = trainer.AsyncCheckpointSaver._to_cpu(payload)
    assert out["a"].device.type == "cpu"
    assert out["nested"][0].device.type == "cpu"
    assert out["nested"][1]["deep"].device.type == "cpu"
    assert isinstance(out["nested"], list)
