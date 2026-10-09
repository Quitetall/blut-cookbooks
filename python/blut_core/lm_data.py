"""Causal-LM data preparation: text rows become fixed-length token blocks.

The generic trainer reads a JSONL dataset as a list of row dicts. A
HuggingFace causal LM cannot consume those rows — it needs integer token ids —
and before this module existed every batch of a `from_pretrained` run raised
inside the model's embedding layer. The trainer swallowed each exception, so a
run in which not one batch trained reported `loss=0.0000`, saved a checkpoint,
and exited 0.

Packing follows the standard causal-LM recipe: tokenize every non-empty text,
join them with the end-of-sequence token, and cut the stream into blocks of
`block_size`. Blocks are all the same length, so no padding is needed and no
block is ever made entirely of ignored labels — which is what turns a
padded-and-masked batch into a NaN loss when a corpus has many blank lines
(wikitext has thousands).

This module has no torch dependency. The trainer turns blocks into tensors in
its collate function, so these functions can be tested without a GPU stack.
"""
from __future__ import annotations

from typing import Any, Iterable, Protocol


class Tokenizer(Protocol):
    """The slice of a HuggingFace tokenizer that packing uses."""

    eos_token_id: int | None

    def __call__(self, texts: list[str], add_special_tokens: bool = ...) -> Any: ...


def is_causal_lm(model: Any) -> bool:
    """True for a HuggingFace `PreTrainedModel`, unwrapping DDP's `.module`.

    Returns False when transformers is not installed: then nothing in the
    process can be a HuggingFace model.
    """
    base = getattr(model, "module", model)
    try:
        from transformers import PreTrainedModel
    except ImportError:
        return False
    return isinstance(base, PreTrainedModel)


def texts_from_rows(rows: Iterable[Any], text_field: str) -> list[str]:
    """The non-blank `text_field` values of `rows`, in order.

    Raises when no row carries a non-blank string there. A causal LM handed a
    dataset without text has nothing to learn from, and training on zero
    tokens is exactly the silent success this module exists to prevent.
    """
    rows = list(rows)
    texts = [
        row[text_field]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get(text_field), str)
        and row[text_field].strip()
    ]
    if not texts:
        raise ValueError(
            f"none of {len(rows)} rows has a non-blank string field "
            f"'{text_field}'; set text_field to the column that holds the text"
        )
    return texts


def pack_token_blocks(
    texts: list[str], tokenizer: Tokenizer, block_size: int
) -> list[list[int]]:
    """Tokenize `texts`, join them with EOS, and cut into `block_size` blocks.

    A trailing partial block is dropped once at least one full block exists —
    the usual trade, since keeping it would need padding. A corpus shorter than
    one block becomes a single short block rather than nothing.
    """
    if block_size < 2:
        raise ValueError(f"block_size must be at least 2, got {block_size}")
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    stream: list[int] = []
    for ids in encoded:
        stream.extend(ids)
        if eos is not None:
            stream.append(eos)
    if len(stream) < 2:
        raise ValueError(
            f"the corpus tokenized to {len(stream)} token(s); "
            "a causal LM needs at least two to predict one"
        )
    n_full = len(stream) // block_size
    if n_full == 0:
        return [stream]
    return [stream[i * block_size : (i + 1) * block_size] for i in range(n_full)]
