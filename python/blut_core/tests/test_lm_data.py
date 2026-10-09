"""Unit tests for causal-LM packing in blut_core.lm_data.

Torch-free by design: a fake tokenizer stands in for a HuggingFace one, so
these run anywhere the package imports.
"""
import pytest

from blut_core.lm_data import pack_token_blocks, texts_from_rows


class _CharTokenizer:
    """One token per character, code point as id; EOS is 0."""

    eos_token_id = 0

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False, "packing must not add BOS/EOS per text"
        return {"input_ids": [[ord(c) for c in t] for t in texts]}


# --- texts_from_rows -----------------------------------------------------------

def test_texts_from_rows_drops_blank_and_non_text_rows():
    rows = [{"text": "ab"}, {"text": "   "}, {"text": ""}, {"other": "x"},
            {"text": 5}, "not a dict", {"text": "cd"}]
    assert texts_from_rows(rows, "text") == ["ab", "cd"]


def test_texts_from_rows_raises_when_nothing_is_text():
    # Training a causal LM on zero tokens is the silent success lm_data
    # exists to prevent: it must be an error, not an empty list.
    with pytest.raises(ValueError, match="text_field"):
        texts_from_rows([{"text": ""}, {"body": "x"}], "text")


def test_texts_from_rows_reads_the_named_field():
    assert texts_from_rows([{"body": "hi"}], "body") == ["hi"]


# --- pack_token_blocks ---------------------------------------------------------

def test_pack_joins_texts_with_eos_and_cuts_full_blocks():
    blocks = pack_token_blocks(["ab", "cd"], _CharTokenizer(), block_size=3)
    # stream: a b EOS c d EOS -> two full blocks of 3
    assert blocks == [[97, 98, 0], [99, 100, 0]]


def test_pack_drops_the_trailing_partial_block():
    blocks = pack_token_blocks(["abcd"], _CharTokenizer(), block_size=2)
    # stream: a b c d EOS (5 tokens) -> two full blocks, lone EOS dropped
    assert blocks == [[97, 98], [99, 100]]
    assert all(len(b) == 2 for b in blocks)


def test_pack_keeps_a_corpus_shorter_than_one_block():
    assert pack_token_blocks(["ab"], _CharTokenizer(), block_size=64) == [[97, 98, 0]]


def test_pack_rejects_a_block_too_small_to_predict_from():
    with pytest.raises(ValueError, match="block_size"):
        pack_token_blocks(["abc"], _CharTokenizer(), block_size=1)


def test_pack_rejects_a_corpus_with_nothing_to_predict():
    class _Empty(_CharTokenizer):
        eos_token_id = None

        def __call__(self, texts, add_special_tokens=True):
            return {"input_ids": [[] for _ in texts]}

    with pytest.raises(ValueError, match="at least two"):
        pack_token_blocks(["x"], _Empty(), block_size=4)
