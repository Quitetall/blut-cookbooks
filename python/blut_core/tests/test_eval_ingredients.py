"""The eval ingredients score what they claim to score.

Pinned defects:

- `loss_eval` required a `loss_fn` the evaluator never passed, so it had never
  run at all;
- `accuracy` compared a causal LM's logits at position t with the token AT t,
  not the token it predicts at t + 1;
- `perplexity` weighted each batch by every label, including the first token
  of each block, which the model never predicts.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from blut_core import build_ingredient  # noqa: E402


def _batches(ids, size):
    return [{"input_ids": ids[i:i + size], "labels": ids[i:i + size].clone()}
            for i in range(0, len(ids), size)]


class _NextTokenOracle(torch.nn.Module):
    """Predicts, at every position, exactly the token that comes next."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, labels=None):
        nxt = torch.roll(input_ids, shifts=-1, dims=1)
        return torch.nn.functional.one_hot(nxt, self.vocab).float() * 10.0


def test_accuracy_scores_a_causal_lm_against_the_next_token(monkeypatch):
    import blut_core.lm_data as lm_data
    monkeypatch.setattr(lm_data, "is_causal_lm", lambda model: True)
    evaluate = build_ingredient("eval", "accuracy", {})
    ids = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
    metrics = evaluate(_NextTokenOracle(8), _batches(ids, 2), device="cpu")
    # A perfect next-token predictor. Unshifted, it would match only where a
    # token repeats its successor: never, in these rows.
    assert metrics["top1_acc"] == 1.0


def _tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(
        n_layer=1, n_head=2, n_embd=16, vocab_size=32, n_positions=16))


def test_loss_eval_runs_with_the_training_loss():
    model = _tiny_gpt2()
    evaluate = build_ingredient("eval", "loss_eval", {})
    loss_fn = build_ingredient("loss", "causal_lm", {})
    ids = torch.randint(0, 32, (4, 8))
    metrics = evaluate(model, _batches(ids, 2), device="cpu", loss_fn=loss_fn)
    assert math.isfinite(metrics["loss"]) and metrics["loss"] > 0


def test_loss_eval_falls_back_to_the_models_own_loss():
    model = _tiny_gpt2()
    evaluate = build_ingredient("eval", "loss_eval", {})
    ids = torch.randint(0, 32, (4, 8))
    with_fn = evaluate(model, _batches(ids, 2), device="cpu",
                       loss_fn=build_ingredient("loss", "causal_lm", {}))
    without = evaluate(model, _batches(ids, 2), device="cpu")
    assert without["loss"] == pytest.approx(with_fn["loss"])


def test_perplexity_is_the_exponential_of_the_per_token_loss():
    model = _tiny_gpt2()
    evaluate = build_ingredient("eval", "perplexity", {})
    ids = torch.randint(0, 32, (3, 8))
    metrics = evaluate(model, _batches(ids, 3), device="cpu")
    assert metrics["perplexity"] == pytest.approx(math.exp(metrics["loss"]))
    with torch.no_grad():
        direct = model(ids, labels=ids).loss.item()
    # One batch holding every row: the per-token mean is the model's own loss.
    assert metrics["loss"] == pytest.approx(direct, rel=1e-5)


def test_perplexity_weights_batches_by_the_tokens_they_predict():
    # Two batches that predict different numbers of tokens: the first has
    # most of its labels ignored. A per-token mean must weight it by the few
    # tokens it scores, not by its full size.
    model = _tiny_gpt2()
    evaluate = build_ingredient("eval", "perplexity", {})
    ids = torch.randint(0, 32, (4, 8))
    sparse = ids[:2].clone()
    sparse[:, 2:] = -100
    batches = [{"input_ids": ids[:2], "labels": sparse},
               {"input_ids": ids[2:], "labels": ids[2:].clone()}]
    metrics = evaluate(model, batches, device="cpu")

    total, count = 0.0, 0
    with torch.no_grad():
        for b in batches:
            logits = model(b["input_ids"]).logits[:, :-1, :]
            targets = b["labels"][:, 1:]
            total += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=-100, reduction="sum").item()
            count += int((targets != -100).sum())
    assert metrics["loss"] == pytest.approx(total / count, rel=1e-5)


def test_eval_ingredients_refuse_an_empty_dataset():
    model = _tiny_gpt2()
    for name in ("loss_eval", "perplexity"):
        evaluate = build_ingredient("eval", name, {})
        with pytest.raises(ValueError):
            evaluate(model, [], device="cpu")
