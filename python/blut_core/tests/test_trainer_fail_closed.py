"""The generic trainer fails closed, and its step ingredients actually train.

Every test here pins a defect that used to pass silently:

- a batch exception was caught and the run reported success with loss 0.0;
- an unreadable dataset was replaced by `[]` and trained on nothing;
- `gradient_accumulation` never called `optimizer.step()`;
- a HuggingFace model was called with its batch dict as one positional
  argument, which no HuggingFace model accepts.
"""
import json

import pytest

torch = pytest.importorskip("torch")

import blut_core.trainer as trainer  # noqa: E402
from blut_core import build_ingredient  # noqa: E402


# --- load_dataset ----------------------------------------------------------------

def test_load_dataset_raises_on_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        trainer.load_dataset({"dataset_path": str(tmp_path / "absent.jsonl")})


def test_load_dataset_raises_on_an_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n")
    with pytest.raises(ValueError, match="no rows"):
        trainer.load_dataset({"dataset_path": str(path)})


def test_load_dataset_names_the_bad_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "ok"}\n{not json}\n')
    with pytest.raises(ValueError, match=":2:"):
        trainer.load_dataset({"dataset_path": str(path)})


def test_load_dataset_reads_jsonl_and_json_arrays(tmp_path):
    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('{"x": 1}\n\n{"x": 2}\n')
    array = tmp_path / "rows.json"
    array.write_text(json.dumps([{"x": 1}, {"x": 2}]))
    assert trainer.load_dataset({"dataset_path": str(jsonl)}) == [{"x": 1}, {"x": 2}]
    assert trainer.load_dataset({"dataset_path": str(array)}) == [{"x": 1}, {"x": 2}]


# --- train_loop ----------------------------------------------------------------

def _config(tmp_path):
    return {
        "dataset_path": "unused",
        "output_dir": str(tmp_path / "out"),
        "optimizer": {"kind": "optimizer", "name": "adamw", "config": {"lr": 1e-3}},
        "scheduler": {"kind": "scheduler", "name": "constant", "config": {}},
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu",
    }


def _ingredients(step_fn):
    return {
        "model": torch.nn.Linear(2, 1),
        "loss": None,
        "step": step_fn,
        "logging": lambda *a, **k: None,
        "checkpoint": lambda payload, path: torch.save(payload, path),
    }


def test_a_failing_batch_fails_the_run(tmp_path):
    def exploding_step(model, optimizer, loss_fn, batch, **kw):
        raise RuntimeError("batch exploded")

    rows = [{"x": i} for i in range(4)]
    with pytest.raises(RuntimeError, match="batch exploded"):
        trainer.train_loop(_config(tmp_path), _ingredients(exploding_step), rows)
    assert not (tmp_path / "out" / "model.pt").exists(), \
        "a run whose batches failed must not leave a checkpoint behind"


def test_a_successful_run_reports_the_mean_step_loss(tmp_path, capsys):
    losses = iter([1.0, 3.0])

    def fixed_step(model, optimizer, loss_fn, batch, **kw):
        return torch.tensor(next(losses))

    rows = [{"x": i} for i in range(4)]  # batch_size 2 -> two steps
    trainer.train_loop(_config(tmp_path), _ingredients(fixed_step), rows)
    assert "loss=2.0000" in capsys.readouterr().out
    assert (tmp_path / "out" / "model.pt").exists()


def test_the_optimizer_holds_the_models_current_parameters(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer, _ = trainer.build_optimization(_config(tmp_path), model)
    held = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert held == {id(p) for p in model.parameters()}


# --- step ingredients ------------------------------------------------------------

class _CountingOptimizer:
    def __init__(self):
        self.steps = 0
        self.zeroed = 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        self.zeroed += 1


def test_gradient_accumulation_steps_the_optimizer_every_n_batches():
    step_fn = build_ingredient(
        "step", "gradient_accumulation", {"accumulation_steps": 3, "max_grad_norm": 0})
    model = torch.nn.Linear(2, 1)
    opt = _CountingOptimizer()

    def loss_fn(output, batch):
        return output.sum()

    for _ in range(7):
        step_fn(model, opt, loss_fn, torch.ones(1, 2))
    # 7 micro-steps at 3 per update: updates after the 3rd and 6th.
    assert opt.steps == 2


def test_gradient_accumulation_rejects_zero_steps():
    with pytest.raises(ValueError, match="accumulation_steps"):
        build_ingredient("step", "gradient_accumulation", {"accumulation_steps": 0})


def test_standard_step_trains_a_huggingface_causal_lm():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=16))
    step_fn = build_ingredient("step", "standard", {})
    loss_fn = build_ingredient("loss", "causal_lm", {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ids = torch.randint(0, 32, (4, 8))
    batch = {"input_ids": ids, "labels": ids.clone()}

    first = last = step_fn(model, optimizer, loss_fn, batch).item()
    for _ in range(20):
        last = step_fn(model, optimizer, loss_fn, batch).item()
    assert last < first, f"loss did not fall: {first} -> {last}"


def test_cross_entropy_names_causal_lm_when_given_a_model_output():
    loss_fn = build_ingredient("loss", "cross_entropy", {})

    class _Output:
        logits = torch.zeros(1, 2)

    with pytest.raises(TypeError, match="causal_lm"):
        loss_fn(_Output(), torch.zeros(1, dtype=torch.long))
