"""Owner-side contract tests for the migrated Python run-ledger surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blut_core.run_ledger import (
    INTENTS,
    LEDGER_FILENAME,
    OUTCOMES,
    SCHEMA,
    TIERS,
    RunLedger,
    data_dir,
    ledger_path,
    track,
)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_default_path_contract_respects_training_data_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LAMU_TRAIN_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert ledger_path() == tmp_path / LEDGER_FILENAME


def test_track_writes_rust_compatible_start_and_end_records(tmp_path: Path) -> None:
    target = tmp_path / LEDGER_FILENAME
    with track(
        "run:7",
        "lamquant_joint_codec",
        tenant="lamquant",
        intent="campaign",
        experiment="E1",
        seed=7,
        path=target,
        trainer_run_id="joint-7",
    ) as ledger:
        ledger.update_identity(config_fingerprint="abc123")
        ledger.end("completed", metrics={"best_val_r": 0.91})

    started, ended = _records(target)
    assert started["schema"] == ended["schema"] == SCHEMA
    assert started["kind"] == "run_started"
    assert started["intent"] in INTENTS
    assert ended["kind"] == "run_ended"
    assert ended["outcome"] in OUTCOMES
    assert ended["tier"] in TIERS
    assert ended["identity"] == {
        "blut_job_id": "run:7",
        "config_fingerprint": "abc123",
        "trainer_run_id": "joint-7",
    }


def test_run_ledger_rejects_unknown_intent_outcome_and_tier(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="intent must be one of"):
        RunLedger("bad", "recipe", intent="unknown", path=tmp_path / "bad.jsonl")

    ledger = RunLedger("run:8", "recipe", path=tmp_path / LEDGER_FILENAME)
    ledger.start()
    with pytest.raises(ValueError, match="outcome must be one of"):
        ledger.end("unknown")
    with pytest.raises(ValueError, match="tier must be one of"):
        ledger.end("completed", tier="invented")
