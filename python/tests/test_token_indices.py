"""`trainable_token_indices` pass-through of the image-text trainer (CPU, tiny random Idefics3, no GPU).

Pins: the job spec reaches peft's LoraConfig; only the named rows (plus LoRA) train; host rows never
move; a saved adapter reloads on a freshly loaded extended base with identical logits; a snapshot that is
not extended, a tied lm_head and out-of-range rows are refused; and the trainable count of the real
SmolVLM-500M shape (analytic) is 5,611,520. The torch tests skip without torch, transformers and peft
(run them with the image-text venv's python).
"""
import sys
from pathlib import Path

import pytest

_PY_DIR = Path(__file__).resolve().parents[2] / "src/backends/hf_trainer/python"
sys.path.insert(0, str(_PY_DIR))
import hf_image_text_train as train  # noqa: E402


# --- real shape, analytic --------------------------------------------------

def lora_params(rank, hidden, kv_dim, layers):
    """q_proj (hidden -> hidden) and v_proj (hidden -> kv_dim) per layer, A and B matrices."""
    return layers * rank * ((hidden + hidden) + (hidden + kv_dim))


def test_real_smolvlm_trainable_count_is_5_611_520():
    hidden, heads, kv_heads, layers = 960, 15, 5, 32
    kv_dim = hidden // heads * kv_heads
    rows, matrices = 51776 - 49280, 2  # ctok/1: 2,496 new rows in embed_tokens and lm_head
    lora = lora_params(8, hidden, kv_dim, layers)
    assert lora == 819_200
    assert rows * hidden * matrices == 4_792_320
    assert lora + rows * hidden * matrices == 5_611_520
    # the rejected full-copy route adds both whole matrices: (49280 + 2496) x 960 x 2 = 99,409,920
    assert 51776 * hidden * 2 == 99_409_920


# --- spec parsing (no torch) -----------------------------------------------

def test_resolve_expands_ranges_and_lists():
    got = train.resolve_token_indices({"embed_tokens": {"start": 5, "end": 8}, "lm_head": [7, 5, 5]})
    assert got == {"embed_tokens": [5, 6, 7], "lm_head": [5, 7]}
    assert train.resolve_token_indices(None) is None and train.resolve_token_indices({}) is None


@pytest.mark.parametrize("bad", [
    {"model.embed_tokens": [1]}, {"lm_head": []}, {"lm_head": {"start": 3, "end": 3}}, {"embed_tokens": [-1, 2]},
])
def test_resolve_refuses_malformed(bad):
    with pytest.raises(ValueError):
        train.resolve_token_indices(bad)


# --- tiny Idefics3 ----------------------------------------------------------

try:
    import peft
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import Idefics3Config, Idefics3ForConditionalGeneration, PreTrainedTokenizerFast
    HAVE_STACK = True
except ImportError:
    HAVE_STACK = False

needs_stack = pytest.mark.skipif(not HAVE_STACK, reason="torch/transformers/peft not installed")

REGEX = r"model\.text_model\.layers\.\d+\.self_attn\.[qv]_proj"
HIDDEN, NEW = 32, 12


def tokenizer(extra=0):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(["a b c d e f g", "the quick brown fox"] * 4, trainers.BpeTrainer(
        vocab_size=300, special_tokens=["<|im_end|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|im_end|>")
    if extra:
        fast.add_tokens([f"<new:{i}>" for i in range(extra)], special_tokens=True)
    return fast


def tiny_model(vocab, tie=False, seed=0):
    torch.manual_seed(seed)
    cfg = Idefics3Config(
        vocab_size=vocab, tie_word_embeddings=tie, scale_factor=1,
        text_config=dict(model_type="llama", vocab_size=vocab, hidden_size=HIDDEN, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         tie_word_embeddings=tie, max_position_embeddings=128),
        vision_config=dict(model_type="idefics3_vision", hidden_size=32, intermediate_size=64,
                           num_hidden_layers=1, num_attention_heads=2, image_size=32, patch_size=16))
    return Idefics3ForConditionalGeneration(cfg).eval()


def extended(host_n, tok, tie=False):
    """What `build-base` writes: host rows byte-identical, new rows initialised (here: seeded noise)."""
    model = tiny_model(host_n, tie)
    model.resize_token_embeddings(len(tok), mean_resizing=False)
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for w in {id(m.weight): m.weight for m in (model.get_input_embeddings(), model.get_output_embeddings())}.values():
            w[host_n:] = torch.randn(w[host_n:].shape, generator=gen)
    return model


SPEC = {"peft": {"method": "lora", "rank": 4, "alpha": 8}}


def spec(**kw):
    return {**SPEC, **kw}


def setup(tie=False):
    host_n = len(tokenizer())
    tok = tokenizer(NEW)
    return host_n, tok, extended(host_n, tok, tie)


@needs_stack
def test_pass_through_trains_only_new_rows_and_lora_with_the_analytic_count():
    host_n, tok, model = setup()
    cfg = train.build_lora_config(peft, spec(trainable_token_indices={
        "embed_tokens": {"start": host_n, "end": host_n + NEW}, "lm_head": {"start": host_n, "end": host_n + NEW}}),
        REGEX, model, tok)
    assert set(cfg.trainable_token_indices) == {"embed_tokens", "lm_head"}
    wrapped = peft.get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    assert trainable == lora_params(4, HIDDEN, 16, 2) + 2 * NEW * HIDDEN
    for name, p in wrapped.named_parameters():
        if p.requires_grad:
            assert "lora_" in name or "trainable_tokens_delta" in name, name


@needs_stack
def test_host_rows_unchanged_and_reload_gives_identical_logits(tmp_path):
    host_n, tok, model = setup()
    host_embed = model.get_input_embeddings().weight.detach().clone()
    host_head = model.get_output_embeddings().weight.detach().clone()
    base_dir, adapter = tmp_path / "base", tmp_path / "adapter"
    model.save_pretrained(base_dir)  # the extended snapshot
    cfg = train.build_lora_config(peft, spec(trainable_token_indices={
        "embed_tokens": {"start": host_n, "end": host_n + NEW}, "lm_head": {"start": host_n, "end": host_n + NEW}}),
        REGEX, model, tok)
    wrapped = peft.get_peft_model(model, cfg)
    params = [p for p in wrapped.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2)
    ids = torch.tensor([[3, 4, 5, host_n, host_n + 3, host_n + NEW - 1, 6]])
    wrapped.train()
    for _ in range(3):
        opt.zero_grad()
        wrapped(input_ids=ids, labels=ids).loss.backward()
        opt.step()
    wrapped.eval()
    with torch.no_grad():
        want = wrapped(input_ids=ids).logits
    wrapped.save_pretrained(adapter)
    from transformers import AutoModelForImageTextToText
    fresh = AutoModelForImageTextToText.from_pretrained(base_dir, dtype=torch.float32)
    got_model = peft.PeftModel.from_pretrained(fresh, adapter).eval()
    with torch.no_grad():
        assert torch.equal(got_model(input_ids=ids).logits, want)
        merged = got_model.merge_and_unload().eval()
        assert torch.allclose(merged(input_ids=ids).logits, want, atol=1e-4, rtol=0)
    embed, head = merged.get_input_embeddings().weight, merged.get_output_embeddings().weight
    assert torch.equal(embed[:host_n], host_embed[:host_n]) and torch.equal(head[:host_n], host_head[:host_n])
    assert not torch.equal(embed[host_n:], host_embed[host_n:])  # the new rows did train


@needs_stack
def test_refuses_a_base_that_is_not_the_extended_snapshot():
    host_n = len(tokenizer())
    tok = tokenizer(NEW)
    with pytest.raises(ValueError, match="not the extended one"):
        train.build_lora_config(peft, spec(trainable_token_indices={"embed_tokens": [host_n]}), REGEX,
                                tiny_model(host_n), tok)


@needs_stack
def test_refuses_out_of_range_rows_and_a_tied_head():
    host_n, tok, model = setup()
    with pytest.raises(ValueError, match=">="):
        train.build_lora_config(peft, spec(trainable_token_indices={"embed_tokens": [host_n + NEW]}), REGEX, model, tok)
    _, _, tied = setup(tie=True)
    with pytest.raises(ValueError, match="tied"):
        train.build_lora_config(peft, spec(trainable_token_indices={"lm_head": [host_n]}), REGEX, tied, tok)


@needs_stack
def test_unset_leaves_the_plain_lora_config():
    host_n, tok, model = setup()
    assert train.build_lora_config(peft, spec(), REGEX, model, tok).trainable_token_indices is None
