"""Windows containment, end to end through BLUT.

Runs the `blut-core` binary on a real Windows host and checks, from outside,
what the Job Object containment does to real training processes:

1. `train`: the `train_from_dataset` recipe trains a tiny HuggingFace causal
   LM on the CPU and evaluates it. Both stage processes (trainer, evaluator)
   must be placed in a Job Object and attached, with the cap and the peak
   reported and no "never attached" warning, and the run must produce its
   checkpoint and evaluation report.
2. `timeout`: a PlanSpec runs `train_model` with a soft timeout on a model
   that starts a grandchild and then hangs. BLUT must time the stage out and
   kill the whole tree: trainer and grandchild.
3. `driver-kill`: the same hanging plan without a timeout; the BLUT driver
   process is killed outright. Closing its Job Object must take the trainer
   and grandchild with it.

Every process the checks expect dead is first confirmed alive, so a probe
that sees nothing cannot pass. Exits non-zero on the first failed check.

Usage: python run.py --blut <path to blut-core.exe> --work <empty dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GIB = 1024 ** 3
# LocalSystemd's default mem_max (44G), the cap every local stage gets.
EXPECTED_CAP = 44 * GIB
BACKEND = "windows-job-object"
NEVER_ATTACHED = "no child was ever assigned"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
TINY_MODEL = "sshleifer/tiny-gpt2"


def fail(message: str) -> None:
    print(f"FAIL: {message}", flush=True)
    sys.exit(1)


def ok(message: str) -> None:
    print(f"ok: {message}", flush=True)


def read_log(path: Path) -> str:
    return ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))


def write_dataset(path: Path, rows: int = 64) -> None:
    words = ["containment", "windows", "job", "object", "training", "blut",
             "memory", "peak", "tree", "kill", "cancel", "timeout"]
    with path.open("w", encoding="utf-8") as handle:
        for i in range(rows):
            text = " ".join(words[(i + k) % len(words)] for k in range(24))
            handle.write(json.dumps({"text": text}) + "\n")


def ingredients(model: dict) -> dict:
    def ing(kind: str, name: str, config: dict | None = None) -> dict:
        return {"kind": kind, "name": name, "config": config or {}}

    return {
        "model": model,
        "optimizer": ing("optimizer", "adamw", {"lr": 1e-3}),
        "scheduler": ing("scheduler", "constant"),
        "loss": ing("loss", "causal_lm"),
        "epochs": 1,
        "batch_size": 4,
        "device": "cpu",
        "max_seq_len": 32,
    }


def check_stage(log: str, unit: str, min_peak: int) -> int:
    """The unit was placed, attached, and exited with a peak. Returns the peak."""
    placed = re.findall(
        rf"containment: {unit} placed by LocalSystemd under {BACKEND}, "
        r"memory cap (\d+) bytes", log)
    if len(placed) != 1:
        fail(f"{unit}: expected one placement line, found {len(placed)}")
    cap = int(placed[0])
    if cap != EXPECTED_CAP:
        fail(f"{unit}: memory cap {cap} bytes, expected {EXPECTED_CAP}")
    if not re.search(rf"containment: {unit} attached to {BACKEND} \(pid \d+\)", log):
        fail(f"{unit}: placed but never reported attached to the Job Object")
    peaks = re.findall(
        rf"containment: {unit} exited; peak memory (\d+) bytes \({BACKEND}\)", log)
    if len(peaks) != 1:
        fail(f"{unit}: expected one peak line, found {len(peaks)}")
    peak = int(peaks[0])
    if peak < min_peak:
        fail(f"{unit}: peak {peak} bytes is below the {min_peak} a torch process commits")
    ok(f"{unit}: attached to the Job Object, cap {cap} bytes, peak {peak} bytes")
    return peak


def check_no_unattached_job(log: str, label: str) -> None:
    if NEVER_ATTACHED in log:
        fail(f"{label}: a run's Job Object never had a child assigned")
    if "UNCONTAINED" in log:
        fail(f"{label}: a stage ran uncontained")
    ok(f"{label}: no 'never attached' or 'UNCONTAINED' warning")


def blut_env(work: Path) -> dict:
    env = dict(os.environ)
    env["LAMU_TRAIN_JOBS_DIR"] = str(work / "jobs")
    env["BLUT_HOME"] = str(work / "blut-home")
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "python"), str(HERE)])
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("BLUT_NO_CONTAIN", None)
    env.pop("BLUT_CONTAINMENT", None)
    return env


def run_train(blut: Path, work: Path, dataset: Path) -> None:
    args = {"dataset_path": str(dataset), "eval_ratio": 0.25,
            **ingredients({"kind": "model", "name": "from_pretrained",
                           "config": {"model_name": TINY_MODEL}})}
    log_path = work / "train.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [str(blut), "recipe", "run", "train_from_dataset", "--args", json.dumps(args)],
            env=blut_env(work), stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    log = read_log(log_path)
    print(log[-4000:])
    if proc.returncode != 0:
        fail(f"train_from_dataset exited {proc.returncode}")
    ok("train_from_dataset exited 0")
    check_stage(log, "blut-train_model", 64 * 1024 * 1024)
    check_stage(log, "blut-evaluator", 64 * 1024 * 1024)
    check_no_unattached_job(log, "train")

    jobs = work / "jobs"
    weights = list(jobs.rglob("checkpoint/model.pt"))
    hf_configs = list(jobs.rglob("checkpoint/hf/config.json"))
    reports = list(jobs.rglob("eval_report.json"))
    if len(weights) != 1 or len(hf_configs) != 1 or len(reports) != 1:
        fail(f"expected one checkpoint and one report, found weights={weights} "
             f"hf={hf_configs} reports={reports}")
    if weights[0].stat().st_size == 0:
        fail(f"{weights[0]} is empty")
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    loss = report.get("loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(loss):
        fail(f"eval report has no finite loss: {report}")
    ok(f"output: {weights[0].stat().st_size} byte checkpoint, held-out loss {loss:.4f}")


def hanging_plan(work: Path, dataset: Path, pid_file: Path, name: str,
                 timeout: dict | None) -> Path:
    train = {"stage": "train_model",
             "args": ingredients({"kind": "model", "name": "custom_module",
                                  "config": {"module_path": "hang_probe.HangingNet",
                                             "init_kwargs": {"pid_file": str(pid_file)}}})}
    if timeout is not None:
        train["timeout"] = timeout
    plan = {"name": name,
            "nodes": [{"stage": "load_dataset", "args": {"path": str(dataset)}}, train],
            "edges": [[0, 1]]}
    path = work / f"{name}.json"
    path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    return path


def start_hanging(blut: Path, work: Path, plan: Path, pid_file: Path, label: str):
    """Start the plan; wait until the trainer and its grandchild exist."""
    log_path = work / f"{label}.log"
    log = log_path.open("w", encoding="utf-8")
    driver = subprocess.Popen([str(blut), "recipe", "declare", str(plan), "--run"],
                              env=blut_env(work), stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 600
    while not pid_file.exists():
        if driver.poll() is not None:
            log.close()
            print(read_log(log_path)[-4000:])
            fail(f"{label}: the driver exited ({driver.returncode}) before the trainer started")
        if time.monotonic() > deadline:
            driver.kill()
            fail(f"{label}: the trainer did not start within 600 s")
        time.sleep(0.5)
    pids = json.loads(pid_file.read_text(encoding="utf-8"))
    tree = [psutil.Process(pids["trainer"]), psutil.Process(pids["grandchild"])]
    # Falsify the probe first: both must be alive before anything is killed.
    for proc in tree:
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            fail(f"{label}: pid {proc.pid} was dead before any teardown")
    ok(f"{label}: trainer {pids['trainer']} and grandchild {pids['grandchild']} are running")
    return driver, log, log_path, tree


def assert_tree_dead(tree, label: str, within: float = 30.0) -> None:
    _, alive = psutil.wait_procs(tree, timeout=within)
    if alive:
        fail(f"{label}: still running {within} s after teardown: {[p.pid for p in alive]}")
    ok(f"{label}: trainer and grandchild are gone")


def run_timeout(blut: Path, work: Path, dataset: Path) -> None:
    pid_file = work / "timeout.pids.json"
    soft = 120
    plan = hanging_plan(work, dataset, pid_file, "windows_containment_timeout",
                        {"soft": {"secs": soft, "nanos": 0},
                         "hard": {"secs": soft + 180, "nanos": 0}})
    driver, log, log_path, tree = start_hanging(blut, work, plan, pid_file, "timeout")
    try:
        code = driver.wait(timeout=soft + 300)
    except subprocess.TimeoutExpired:
        driver.kill()
        fail("timeout: the driver did not finish after the stage timeout")
    log.close()
    text = read_log(log_path)
    print(text[-4000:])
    if code == 0:
        fail("timeout: a timed-out plan exited 0")
    ok(f"timeout: the plan failed as it should (exit {code})")
    if not re.search(rf"containment: blut-train_model cancelled; killing its process "
                     rf"tree \({BACKEND}\)", text):
        fail("timeout: no tree kill was logged for the timed-out stage")
    if "timed out" not in text:
        fail("timeout: the failure does not say the stage timed out")
    ok("timeout: the soft timeout cancelled the stage and killed its tree")
    if not re.search(rf"containment: blut-train_model attached to {BACKEND}", text):
        fail("timeout: the trainer was never attached to its Job Object")
    check_no_unattached_job(text, "timeout")
    assert_tree_dead(tree, "timeout")


def run_driver_kill(blut: Path, work: Path, dataset: Path) -> None:
    pid_file = work / "driver-kill.pids.json"
    plan = hanging_plan(work, dataset, pid_file, "windows_containment_driver_kill", None)
    driver, log, log_path, tree = start_hanging(blut, work, plan, pid_file, "driver-kill")
    psutil.Process(driver.pid).kill()  # TerminateProcess: no cleanup runs
    driver.wait(timeout=60)
    log.close()
    text = read_log(log_path)
    print(text[-2000:])
    if not re.search(rf"containment: blut-train_model attached to {BACKEND}", text):
        fail("driver-kill: the trainer was never attached to its Job Object")
    ok("driver-kill: the BLUT driver was killed outright")
    assert_tree_dead(tree, "driver-kill")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blut", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != "win32":
        fail("this check runs on Windows only")
    args.work.mkdir(parents=True, exist_ok=True)
    dataset = args.work / "train.jsonl"
    write_dataset(dataset)

    started = time.monotonic()
    run_train(args.blut, args.work, dataset)
    run_timeout(args.blut, args.work, dataset)
    run_driver_kill(args.blut, args.work, dataset)
    print(f"ALL CHECKS PASSED in {time.monotonic() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
