"""Image-text trainer: streaming, masking, resume.

The data tests need only the standard library. The masking-equality and
GPU tests need a training stack and the pinned SmolVLM snapshot and skip
without them:

    BLUT_IMAGE_TEXT_PYTHON  python of the image-text venv (transformers 5.x)
    BLUT_SMOLVLM_DIR        local pinned SmolVLM-500M-Instruct snapshot
"""
import importlib.util
import json
import os
import random
import signal
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import pytest

_PY_DIR = Path(__file__).resolve().parents[2] / "src/backends/hf_trainer/python"
sys.path.insert(0, str(_PY_DIR))
import hf_image_text_data as data  # noqa: E402

SMOLVLM = os.environ.get("BLUT_SMOLVLM_DIR", "")
VENV_PY = os.environ.get("BLUT_IMAGE_TEXT_PYTHON", "")


def write_shards(tmp_path, n, shards=3, name="s"):
    paths = [tmp_path / f"{name}{i}.jsonl" for i in range(shards)]
    files = [p.open("w") for p in paths]
    for i in range(n):
        files[i * shards // n].write(json.dumps({"prompt": f"p{i}", "completion": f"c{i}"}) + "\n")
    for f in files:
        f.close()
    return paths


def ids(stream, k=None):
    out = []
    while k is None or len(out) < k:
        line = stream.next_line()
        if line is None:
            break
        out.append(json.loads(line)["prompt"])
    return out


# ---- streaming ------------------------------------------------------------


def test_epoch_visits_every_row_once_in_shuffled_order(tmp_path):
    shards = write_shards(tmp_path, 1000)
    got = ids(data.RowStream(shards, seed=1, buffer=64, epochs=1))
    assert sorted(got) == sorted(f"p{i}" for i in range(1000))
    assert got != [f"p{i}" for i in range(1000)]  # actually shuffled


def test_stream_is_deterministic_and_seed_dependent(tmp_path):
    shards = write_shards(tmp_path, 500)
    a = ids(data.RowStream(shards, 7, 32, 2))
    b = ids(data.RowStream(shards, 7, 32, 2))
    c = ids(data.RowStream(shards, 8, 32, 2))
    assert a == b and a != c
    assert len(a) == 1000 and a[:500] != a[500:]  # epochs reshuffle


def test_buffer_of_one_keeps_shard_order(tmp_path):
    shards = write_shards(tmp_path, 100)
    assert ids(data.RowStream(shards, 1, 1, 1)) == [f"p{i}" for i in range(100)]


@pytest.mark.parametrize("skip", [0, 1, 63, 64, 65, 499, 500, 777, 1000])
def test_resume_position_continues_the_same_sequence(tmp_path, skip):
    shards = write_shards(tmp_path, 500)
    full = ids(data.RowStream(shards, 3, 64, 2))
    resumed = data.RowStream(shards, 3, 64, 2, skip=skip)
    assert resumed.consumed == skip
    assert ids(resumed) == full[skip:]


def test_streaming_memory_does_not_grow_with_dataset_size(tmp_path):
    # 210k rows (the real size): peak Python allocation while streaming the
    # whole set must stay near the shuffle buffer's size, not the dataset's.
    shards = write_shards(tmp_path, 210_000, shards=4)
    tracemalloc.start()
    s = data.RowStream(shards, 1, 1000, 1)
    n = 0
    while s.next_line() is not None:
        n += 1
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    size = sum(p.stat().st_size for p in shards)
    assert n == 210_000
    assert data.count_rows(shards) == 210_000
    assert peak < size / 20, f"peak {peak} bytes vs dataset {size}"
    assert peak < 2_000_000


# ---- masking (no tokenizer needed) -----------------------------------------


def test_collate_masks_prompt_and_padding():
    batch = [([1, 2], [1, 2, 3, 4]), ([5], [5, 6])]
    c = data.collate(batch, pad_id=0)
    assert c["input_ids"] == [[1, 2, 3, 4], [5, 6, 0, 0]]
    assert c["attention_mask"] == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert c["labels"] == [[-100, -100, 3, 4], [-100, 6, -100, -100]]


def test_rows_need_prompt_and_completion():
    with pytest.raises(ValueError, match="prompt"):
        data.row_texts({"packet": {}, "target": "e2e4"})


# ---- equality with the KnightLM reference harness --------------------------

# Verbatim from KnightLM model/train/harness.py (`encode` and the batch
# builder inside `train`), with the row->prompt step made explicit.
def _reference_encode(row, processor):
    tokenizer = processor.tokenizer
    packet = json.dumps(row["packet"], sort_keys=True, separators=(",", ":"))
    chat = [{"role": "user", "content": [{"type": "text", "text": (
        "Choose a legal chess move. Reply with just the UCI move.\n" + packet
    )}]}]
    prompt = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    chat.append({"role": "assistant", "content": [{"type": "text", "text": row["target"]}]})
    answer = processor.apply_chat_template(chat, tokenize=False)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids_ = tokenizer.encode(answer, add_special_tokens=False)
    if ids_[:len(prompt_ids)] != prompt_ids:
        raise ValueError("SmolVLM prompt and completion tokenize to different prefixes")
    end = ids_.index(tokenizer.eos_token_id, len(prompt_ids)) + 1
    return prompt_ids, ids_[:end]


def _reference_tensors(batch, tokenizer):
    sequences = [ids_ for _, ids_ in batch]
    width = max(map(len, sequences))
    input_ids, attention_mask, labels = [], [], []
    for (prefix, _), ids_ in zip(batch, sequences, strict=True):
        pad = [tokenizer.pad_token_id] * (width - len(ids_))
        input_ids.append(ids_ + pad)
        attention_mask.append([1] * len(ids_) + [0] * len(pad))
        labels.append([-100] * len(prefix) + ids_[len(prefix):] + [-100] * len(pad))
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@pytest.mark.skipif(not SMOLVLM, reason="BLUT_SMOLVLM_DIR not set")
def test_encoding_and_labels_equal_the_knightlm_harness():
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(SMOLVLM, local_files_only=True)
    rng = random.Random(0)
    rows = []
    for i in range(40):
        packet = {"fen": f"fen-{i}-" + "x" * rng.randint(0, 50),
                  "legal_moves": ["e2e4", "d2d4", "g1f3"][: 1 + i % 3], "n": i}
        rows.append({"packet": packet, "target": rng.choice(packet["legal_moves"])})
    ours, theirs = [], []
    for row in rows:
        prompt = ("Choose a legal chess move. Reply with just the UCI move.\n"
                  + json.dumps(row["packet"], sort_keys=True, separators=(",", ":")))
        ours.append(data.encode(prompt, row["target"], processor))
        theirs.append(_reference_encode(row, processor))
    assert ours == theirs
    for lo in range(0, 40, 8):
        assert (data.collate(ours[lo:lo + 8], processor.tokenizer.pad_token_id)
                == _reference_tensors(theirs[lo:lo + 8], processor.tokenizer))
    # loss only on the completion: every row has trained tokens, none on the prompt
    c = data.collate(ours[:8], processor.tokenizer.pad_token_id)
    for (p, _), lab in zip(ours[:8], c["labels"]):
        assert set(lab[:len(p)]) == {-100} and any(t != -100 for t in lab)


# ---- GPU: run through the wrapper, kill, resume -----------------------------

gpu = pytest.mark.skipif(not (SMOLVLM and VENV_PY), reason="needs BLUT_SMOLVLM_DIR and BLUT_IMAGE_TEXT_PYTHON")


def _job(tmp, out, steps, **extra):
    shards = write_shards(tmp, 256, shards=2, name="g")
    spec = {
        "task": "sft", "arch": "image_text", "base_model": SMOLVLM,
        "train_dataset_path": str(shards[0]), "dataset_shards": [str(s) for s in shards],
        "output_dir": str(out), "lr": 2e-4, "epochs": 1, "batch_size": 4, "grad_accum": 2,
        "seq_len": 512, "seed": 42, "extra": {}, "peft": {"method": "lora", "rank": 8, "alpha": 16},
        "max_steps": steps, "save_steps": 5, "shuffle_buffer": 64, "logging_steps": 1,
        "verify_reload": False,
    }
    spec.update(extra)
    path = Path(str(out) + ".job.json")
    path.write_text(json.dumps(spec))
    return path


WRAPPER = _PY_DIR / "hf_trainer_runner.py"


def _events(text):
    return [json.loads(l) for l in text.splitlines() if l.startswith("{")]


def _losses(events):
    return {e["step"]: e["loss"] for e in events if e["kind"] == "step"}


@gpu
def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    import torch  # noqa: F401  (skips below if the host has no torch)
    from safetensors.torch import load_file

    full_out, cut_out = tmp_path / "full", tmp_path / "cut"
    r = subprocess.run([VENV_PY, str(WRAPPER), str(_job(tmp_path, full_out, 20))],
                       capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    full = _losses(_events(r.stdout))
    assert sorted(full) == list(range(1, 21))

    # Start the same job, kill it hard once the step-10 checkpoint is published.
    job = _job(tmp_path, cut_out, 20)
    p = subprocess.Popen([VENV_PY, str(WRAPPER), str(job)], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True)
    saw = 0
    for line in p.stdout:
        if line.startswith('{"kind": "saved"'):
            saw += 1
            if saw == 2:  # step 5 and step 10
                break
    p.send_signal(signal.SIGKILL)
    p.wait()
    ckpts = sorted((cut_out / "checkpoints").glob("step-*"))
    assert ckpts, "no checkpoint survived the kill"

    job = _job(tmp_path, cut_out, 20, resume=True)
    r = subprocess.run([VENV_PY, str(WRAPPER), str(job)], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-2000:]
    ev = _events(r.stdout)
    resumed = [e for e in ev if e["kind"] == "resumed"]
    assert resumed and resumed[0]["step"] >= 10
    after = _losses(ev)
    assert min(after) == resumed[0]["step"] + 1 and max(after) == 20
    for s, v in after.items():
        assert v == pytest.approx(full[s], rel=1e-3, abs=1e-4), (s, v, full[s])
    a = load_file(full_out / "adapter_model.safetensors")
    b = load_file(cut_out / "adapter_model.safetensors")
    assert a.keys() == b.keys()
    for k in a:
        assert torch.allclose(a[k], b[k], rtol=1e-3, atol=1e-5), k
