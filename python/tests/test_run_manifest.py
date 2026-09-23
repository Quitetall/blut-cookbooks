"""Unit tests for blut_core.run_manifest — Phase 1 of the 100% roadmap.

Targets the pure-logic surface: ULID, config hashing, checkpoint
hashing, manifest dataclass round-trips, atomic write, context-manager
protocol (clean exit + exception path).

Excluded:
  - `_library_versions` / `_hardware_info` — environment-dependent;
    smoke-tested only.
  - `_git_sha` — depends on git state; smoke-tested only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from blut_core import run_manifest
from blut_core.run_manifest import (
    RunManifest,
    _checkpoint_sha,
    _config_sha,
    _git_sha,
    _hardware_info,
    _library_versions,
    _ulid,
)



# ============================================================
# Internal helpers
# ============================================================


class TestUlid:

    def test_returns_hex_string(self) -> None:
        u = _ulid()
        assert isinstance(u, str)
        assert len(u) == 32
        assert all(c in "0123456789abcdef" for c in u)

    def test_unique_across_calls(self) -> None:
        ids = {_ulid() for _ in range(100)}
        assert len(ids) == 100, "ULID collisions in 100 draws — uuid4 broken"


class TestConfigSha:

    def test_deterministic_on_same_dict(self) -> None:
        cfg = {"lr": 1e-3, "batch": 32, "model": "encoder"}
        assert _config_sha(cfg) == _config_sha(cfg)

    def test_sensitive_to_value_change(self) -> None:
        cfg_a = {"lr": 1e-3}
        cfg_b = {"lr": 1e-4}
        assert _config_sha(cfg_a) != _config_sha(cfg_b)

    def test_insensitive_to_key_order(self) -> None:
        cfg_a = {"a": 1, "b": 2, "c": 3}
        cfg_b = {"c": 3, "a": 1, "b": 2}
        assert _config_sha(cfg_a) == _config_sha(cfg_b)

    def test_accepts_list(self) -> None:
        sha = _config_sha([1, 2, 3])
        assert len(sha) == 64

    def test_wraps_non_dict_non_list(self) -> None:
        # Falls back to {"raw": str(x)} per the implementation.
        sha = _config_sha("not-a-dict-or-list")
        assert len(sha) == 64

    def test_format_is_hex_lowercase(self) -> None:
        sha = _config_sha({"x": 1})
        assert sha == sha.lower()
        assert all(c in "0123456789abcdef" for c in sha)


class TestCheckpointSha:

    def test_matches_hashlib_reference(self, tmp_path: Path) -> None:
        f = tmp_path / "ckpt.bin"
        payload = b"weights..." * 4096
        f.write_bytes(payload)
        assert _checkpoint_sha(f) == hashlib.sha256(payload).hexdigest()

    def test_handles_chunked_streaming(self, tmp_path: Path) -> None:
        f = tmp_path / "ckpt.bin"
        # > 1 MB exercises multi-chunk read.
        f.write_bytes(b"x" * (1 << 20) * 2 + b"y" * 1000)
        sha = _checkpoint_sha(f)
        assert len(sha) == 64


class TestEnvironmentHelpers:
    """Smoke tests — exact values are environment-dependent."""

    def test_git_sha_returns_str(self) -> None:
        out = _git_sha()
        assert isinstance(out, str)
        # Either a 40-char SHA or the explicit unknown sentinel.
        assert len(out) == 40 or out == "unknown"

    def test_library_versions_includes_python(self) -> None:
        v = _library_versions()
        assert "python" in v
        assert v["python"].count(".") >= 1  # major.minor at minimum

    def test_hardware_info_includes_hostname_and_platform(self) -> None:
        info = _hardware_info()
        assert "hostname" in info
        assert "platform" in info
        assert "machine" in info


# ============================================================
# RunManifest — start / set_checkpoint / add_dataset / close
# ============================================================


class TestRunManifestLifecycle:

    def test_start_creates_out_dir_and_writes_manifest(
        self, tmp_path: Path,
    ) -> None:
        m = RunManifest.start(
            out_dir=tmp_path, argv=["train.py", "--lr", "1e-3"],
            seed=42, config={"lr": 1e-3},
        )
        assert m.out_dir.exists()
        assert m.out_dir.is_dir()
        manifest_path = m.out_dir / "RUN_MANIFEST.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["run_id"] == m.run_id
        assert data["argv"] == ["train.py", "--lr", "1e-3"]
        assert data["seed"] == 42
        assert data["config_hash"] == _config_sha({"lr": 1e-3})

    def test_start_records_no_config_when_omitted(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["x"])
        assert m.config_hash == "no-config"

    def test_set_checkpoint_records_sha(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["train"], seed=0)
        ckpt = tmp_path / "weights.ckpt"
        ckpt.write_bytes(b"some-weights")
        m.set_checkpoint(ckpt)
        assert m.checkpoint_path == str(ckpt)
        assert m.checkpoint_sha256 == hashlib.sha256(b"some-weights").hexdigest()

    def test_set_checkpoint_handles_missing_file(self, tmp_path: Path) -> None:
        # Set path but don't hash if file doesn't exist (callers may pass
        # the intended path before the file is fully written).
        m = RunManifest.start(out_dir=tmp_path, argv=["train"])
        m.set_checkpoint(tmp_path / "not-yet-written.ckpt")
        assert m.checkpoint_path == str(tmp_path / "not-yet-written.ckpt")
        assert m.checkpoint_sha256 is None

    def test_add_dataset_appends(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["train"])
        m.add_dataset("tusz_v2.0.6", "abc123", "data/manifest.json")
        m.add_dataset("tueg_v3", "def456")
        assert len(m.dataset_manifests) == 2
        assert m.dataset_manifests[0]["name"] == "tusz_v2.0.6"
        assert m.dataset_manifests[0]["sha256"] == "abc123"
        assert m.dataset_manifests[1]["manifest_path"] == ""

    def test_close_records_ended_at_and_exit_code(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["train"])
        assert m.ended_at is None
        m.close(exit_code=0)
        assert m.ended_at is not None
        assert m.exit_code == 0

    def test_close_records_exception(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["train"])
        m.close(exit_code=1, exception="RuntimeError: dataset corrupted")
        assert m.exit_code == 1
        assert m.exception == "RuntimeError: dataset corrupted"


# ============================================================
# Context-manager protocol
# ============================================================


class TestContextManager:

    def test_clean_exit_sets_exit_code_zero(self, tmp_path: Path) -> None:
        with RunManifest.start(out_dir=tmp_path, argv=["x"]) as m:
            pass
        assert m.exit_code == 0
        assert m.exception is None

    def test_exception_records_traceback_and_reraises(self, tmp_path: Path) -> None:
        # Bug in user code → manifest captures the failure but doesn't
        # swallow the exception (returns False from __exit__).
        captured = None
        with pytest.raises(RuntimeError, match="boom"):
            with RunManifest.start(out_dir=tmp_path, argv=["x"]) as m:
                captured = m
                raise RuntimeError("boom")
        assert captured is not None
        assert captured.exit_code == 1
        assert captured.exception is not None
        assert "RuntimeError" in captured.exception
        assert "boom" in captured.exception

    def test_manifest_written_to_disk_on_exception(self, tmp_path: Path) -> None:
        captured = None
        with pytest.raises(ValueError):
            with RunManifest.start(out_dir=tmp_path, argv=["x"]) as m:
                captured = m
                raise ValueError("expected")
        manifest_path = captured.out_dir / "RUN_MANIFEST.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data["exit_code"] == 1
        assert "ValueError" in data["exception"]


# ============================================================
# Atomic write
# ============================================================


class TestAtomicWrite:

    def test_tmp_file_not_left_behind(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["x"])
        m.close(exit_code=0)
        tmp_file = m.out_dir / "RUN_MANIFEST.json.tmp"
        assert not tmp_file.exists(), "atomic-write tmp file leaked"

    def test_multiple_writes_overwrite_cleanly(self, tmp_path: Path) -> None:
        m = RunManifest.start(out_dir=tmp_path, argv=["x"], seed=1)
        m.add_dataset("d1", "sha1")
        m.add_dataset("d2", "sha2")
        m.close(exit_code=0)

        data = json.loads((m.out_dir / "RUN_MANIFEST.json").read_text())
        assert len(data["dataset_manifests"]) == 2
        assert data["dataset_manifests"][0]["name"] == "d1"
        assert data["dataset_manifests"][1]["name"] == "d2"

    def test_out_dir_serialized_as_str(self, tmp_path: Path) -> None:
        # Path() can't go through json directly — the writer converts to str.
        m = RunManifest.start(out_dir=tmp_path, argv=["x"])
        data = json.loads((m.out_dir / "RUN_MANIFEST.json").read_text())
        assert data["out_dir"] == str(m.out_dir)
        assert isinstance(data["out_dir"], str)
