from __future__ import annotations

import sys
import types

import pytest

from blut_core.load_dataset import materialize


class FakeDataset(list):
    def select(self, indices):
        return FakeDataset(self[index] for index in indices)


def _install_fake(monkeypatch, rows):
    module = types.ModuleType("datasets")
    module.load_dataset = lambda name, subset, split: FakeDataset(rows)
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_materializes_stable_jsonl(monkeypatch, tmp_path):
    _install_fake(monkeypatch, [{"z": 1, "a": "x"}, {"a": "y"}])
    output = tmp_path / "dataset.jsonl"

    count = materialize("fixture", "train", output)

    assert count == 2
    assert output.read_text() == '{"a": "x", "z": 1}\n{"a": "y"}\n'


def test_empty_dataset_fails_without_output(monkeypatch, tmp_path):
    _install_fake(monkeypatch, [])
    output = tmp_path / "dataset.jsonl"

    with pytest.raises(ValueError, match="is empty"):
        materialize("fixture", "train", output)

    assert not output.exists()
