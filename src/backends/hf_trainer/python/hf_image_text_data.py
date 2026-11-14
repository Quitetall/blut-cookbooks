"""Streaming data path for the image-text (`arch == "image_text"`) trainer.

Pure standard library: no torch or transformers at module scope, so the
streaming, shuffling and resume logic is unit-testable without a training
stack. Tokenisation (`encode`) takes an already-built processor.

Rows are JSONL objects `{"prompt": str, "completion": str}`. The loss covers
the completion only: `labels` is -100 over the chat-templated prompt (with
the generation prompt) and over padding.

Streaming contract
------------------
`RowStream` reads the shard files in the order given, line by line, through a
fixed-size shuffle buffer seeded from `(seed, epoch)`. Memory is bounded by
`shuffle_buffer` raw lines, whatever the dataset size. The stream is a pure
function of (shards, seed, buffer, epochs), so its position is just the
number of rows already consumed: a resumed run rebuilds the stream and skips
that many rows (reading and shuffling, never parsing or tokenising them).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple


def count_rows(shards: Sequence[Path]) -> int:
    """Non-empty lines across all shards, read in constant memory."""
    n = 0
    for shard in shards:
        with open(shard, "rb") as fh:
            for line in fh:
                if line.strip():
                    n += 1
    return n


def _epoch_rng(seed: int, epoch: int) -> random.Random:
    # Distinct, reproducible stream per epoch. Integer seeding is stable
    # across Python versions for random.Random.
    return random.Random(seed * 1_000_003 + epoch)


def iter_epoch(shards: Sequence[Path], seed: int, epoch: int, buffer: int) -> Iterator[str]:
    """One pass over the shards in order through a seeded shuffle buffer."""
    rng = _epoch_rng(seed, epoch)
    buf: List[str] = []
    for shard in shards:
        with open(shard, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                if buffer <= 1:
                    yield line
                elif len(buf) < buffer:
                    buf.append(line)
                else:
                    i = rng.randrange(buffer)
                    out, buf[i] = buf[i], line
                    yield out
    rng.shuffle(buf)
    yield from buf


class RowStream:
    """Raw JSONL lines over `epochs` epochs, resumable by position."""

    def __init__(self, shards: Sequence[Path], seed: int, buffer: int, epochs: int,
                 skip: int = 0) -> None:
        self.consumed = 0
        self._it = self._gen(list(shards), seed, buffer, epochs)
        for _ in range(skip):
            if next(self._it, None) is None:
                break
            self.consumed += 1

    @staticmethod
    def _gen(shards, seed, buffer, epochs) -> Iterator[str]:
        for epoch in range(epochs):
            yield from iter_epoch(shards, seed, epoch, buffer)

    def next_line(self) -> str | None:
        line = next(self._it, None)
        if line is not None:
            self.consumed += 1
        return line


def row_texts(row: Dict[str, Any]) -> Tuple[str, str]:
    if "prompt" not in row or "completion" not in row:
        raise ValueError(
            "image_text rows must be {\"prompt\": str, \"completion\": str}; "
            f"got keys {sorted(row)}"
        )
    return str(row["prompt"]), str(row["completion"])


def _end_token_ids(tokenizer) -> List[int]:
    """Token ids that close an assistant turn, for this tokenizer."""
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in ("<end_of_utterance>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != unk and tid not in ids:
            ids.append(tid)
    return ids


def encode(prompt: str, completion: str, processor) -> Tuple[List[int], List[int]]:
    """(prompt_ids, full_ids): chat template, generation prompt, completion + end.

    For the pinned SmolVLM this is token-for-token the KnightLM reference
    harness's `encode`: the full sequence ends right after the first end-of-turn
    token that follows the prompt, so trailing template text is not trained on.
    """
    tokenizer = processor.tokenizer
    # Idefics3 keeps the template on the processor; Qwen3.5's processor has
    # none and the template lives on its tokenizer.
    templater = processor if getattr(processor, "chat_template", None) else tokenizer
    chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    prompt_text = templater.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    chat.append({"role": "assistant", "content": [{"type": "text", "text": completion}]})
    answer_text = templater.apply_chat_template(chat, tokenize=False)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    ids = tokenizer.encode(answer_text, add_special_tokens=False)
    if ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("prompt and completion tokenize to different prefixes")
    ends = _end_token_ids(tokenizer)
    cut = [i for i in range(len(prompt_ids), len(ids)) if ids[i] in ends]
    end = cut[0] + 1 if cut else len(ids)
    return prompt_ids, ids[:end]


def collate(batch: Sequence[Tuple[List[int], List[int]]], pad_id: int) -> Dict[str, List[List[int]]]:
    """Right-pad to the longest sequence. Labels: -100 over prompt and padding."""
    width = max(len(ids) for _, ids in batch)
    input_ids, attention_mask, labels = [], [], []
    for prompt_ids, ids in batch:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append([-100] * len(prompt_ids) + ids[len(prompt_ids):] + [-100] * pad)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class Batcher:
    """Encodes rows from a `RowStream` into micro-batches.

    Rows longer than `seq_len` after encoding are dropped and counted (they are
    consumed from the stream, so the position stays a function of rows read).
    A batch is only returned when full; a short tail at end of data is dropped
    so every optimizer step sees complete accumulation windows.
    """

    def __init__(self, stream: RowStream, processor, batch_size: int, seq_len: int) -> None:
        self.stream = stream
        self.processor = processor
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dropped_long = 0

    def next_batch(self):
        batch = []
        while len(batch) < self.batch_size:
            line = self.stream.next_line()
            if line is None:
                return None
            prompt, completion = row_texts(json.loads(line))
            prompt_ids, ids = encode(prompt, completion, self.processor)
            if len(ids) > self.seq_len or len(ids) == len(prompt_ids):
                self.dropped_long += 1
                continue
            batch.append((prompt_ids, ids))
        return batch
