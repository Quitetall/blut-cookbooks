"""Linux containment, end to end through BLUT.

Runs the `blut-core` binary on a Linux host and checks, from outside, what
the engine's containment does to a stage process. The trainer is
`stub_trainer.py`, put in place of `python -m blut_core.trainer` with
`BLUT_PYTHON`, so no model is trained. Checks:

1. `contained`: a `load_dataset` -> `train_model` plan runs under the
   expected backend (default `systemd`) with the launcher's memory cap and a
   reported peak; the trainer sees a variable from the driver's environment;
   the stage leaves its checkpoint.
2. `timeout`: the same plan with a soft stage timeout on a trainer that
   starts a grandchild and hangs. BLUT must stop the whole tree.
3. `sigterm`: the hanging plan without a timeout; the driver gets SIGTERM
   (what `blut cancel` sends). The whole tree must stop.
4. `no-contain`: `BLUT_NO_CONTAIN=1` runs the stage bare, and the placement
   line says so.
5. `no-bus`: with the user bus hidden the stage still runs, under a
   fallback backend the placement line names.
6. No `blut-*` systemd unit is left behind.

Every process a check expects dead is first confirmed alive. Exits non-zero
on the first failed check.

Usage: python linux.py --blut <path to blut-core> --work <empty dir>
       [--expect systemd|cgroup2|rlimit|bare]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
GIB = 1024 ** 3
MIB = 1024 ** 2
EXPECTED_CAP = 44 * GIB  # LocalSystemd's default mem_max
ANSI = re.compile(r"\x1b\[[0-9;]*m")
UNIT = r"blut-train_model-\d+-\d+"


def fail(message: str) -> None:
    print(f"FAIL: {message}", flush=True)
    sys.exit(1)


def ok(message: str) -> None:
    print(f"ok: {message}", flush=True)


def read_log(path: Path) -> str:
    return ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))


def blut_env(work: Path, **extra: str) -> dict:
    env = dict(os.environ)
    env["LAMU_TRAIN_JOBS_DIR"] = str(work / "jobs")
    env["BLUT_HOME"] = str(work / "blut-home")
    env["BLUT_PYTHON"] = str(HERE / "stub_trainer.py")
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("BLUT_NO_CONTAIN", "BLUT_CONTAINMENT", "BLUT_STUB_HANG"):
        env.pop(key, None)
    env.update(extra)
    return env


def plan(work: Path, dataset: Path, name: str, timeout: dict | None) -> Path:
    ing = lambda kind, n: {"kind": kind, "name": n, "config": {}}  # noqa: E731
    train = {"stage": "train_model",
             "args": {"model": ing("model", "custom_module"),
                      "optimizer": ing("optimizer", "adamw"),
                      "scheduler": ing("scheduler", "constant"),
                      "loss": ing("loss", "mse"),
                      "epochs": 1, "batch_size": 4, "device": "cpu"}}
    if timeout is not None:
        train["timeout"] = timeout
    spec = {"name": name,
            "nodes": [{"stage": "load_dataset", "args": {"path": str(dataset)}}, train],
            "edges": [[0, 1]]}
    path = work / f"{name}.json"
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


def run_plan(blut: Path, plan_path: Path, env: dict, log_path: Path, timeout: int = 300):
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([str(blut), "recipe", "declare", str(plan_path), "--run"],
                              env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode, read_log(log_path)


def placement(text: str) -> tuple[str, str]:
    found = re.findall(rf"containment: {UNIT} placed by LocalSystemd under (\S+), (.*)", text)
    if len(found) != 1:
        fail(f"expected one placement line for train_model, found {found}")
    return found[0][0], found[0][1]


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers; read its state.
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def wait_dead(pids: list[int], label: str, within: float = 30.0) -> None:
    deadline = time.monotonic() + within
    while any(alive(p) for p in pids):
        if time.monotonic() > deadline:
            fail(f"{label}: still running {within} s after teardown: "
                 f"{[p for p in pids if alive(p)]}")
        time.sleep(0.2)
    ok(f"{label}: trainer and grandchild are gone")


def unit_active(text: str) -> list[str]:
    units = set(re.findall(rf"containment: ({UNIT}) placed", text))
    active = []
    for unit in units:
        state = subprocess.run(["systemctl", "--user", "is-active", f"{unit}.service"],
                               capture_output=True, text=True).stdout.strip()
        if state in ("active", "activating", "deactivating"):
            active.append(unit)
    return active


def check_contained(blut: Path, work: Path, dataset: Path, expect: str) -> None:
    tag = uuid.uuid4().hex[:12]
    code, text = run_plan(blut, plan(work, dataset, "contained_ok", None),
                          blut_env(work, BLUT_STUB_TAG=tag), work / "contained.log")
    print(text[-3000:])
    if code != 0:
        fail(f"contained: the plan exited {code}")
    kind, applied = placement(text)
    if kind != expect:
        fail(f"contained: placed under {kind}, expected {expect}")
    if expect in ("systemd", "cgroup2"):
        cap = re.search(r"memory cap (\d+) bytes", applied)
        if not cap or int(cap.group(1)) != EXPECTED_CAP:
            fail(f"contained: placement says {applied!r}, expected memory cap {EXPECTED_CAP} bytes")
        peak = re.findall(rf"containment: {UNIT} exited; peak memory (\d+) bytes \({kind}\)", text)
        if len(peak) != 1 or int(peak[0]) < 64 * MIB:
            fail(f"contained: expected one peak of at least 64 MiB, found {peak}")
        ok(f"contained: placed under {kind}, memory cap {EXPECTED_CAP} bytes, peak {peak[0]} bytes")
    else:
        ok(f"contained: placed under {kind} ({applied})")
    if not re.search(rf"containment: {UNIT} attached to {kind} \(pid \d+\)", text):
        fail("contained: placed but never reported attached")
    if f"tag {tag}" not in text:
        fail("contained: the trainer did not see the driver's BLUT_STUB_TAG")
    ok("contained: the trainer saw a variable from the driver's environment")
    weights = list((work / "jobs").rglob("checkpoint/model.pt"))
    if not weights:
        fail("contained: the stage left no checkpoint")
    ok(f"contained: checkpoint at {weights[0].relative_to(work)}")


def start_hanging(blut: Path, work: Path, dataset: Path, name: str, timeout: dict | None):
    pid_file = work / f"{name}.pids.json"
    log_path = work / f"{name}.log"
    log = log_path.open("w", encoding="utf-8")
    driver = subprocess.Popen(
        [str(blut), "recipe", "declare", str(plan(work, dataset, name, timeout)), "--run"],
        env=blut_env(work, BLUT_STUB_HANG=str(pid_file)), stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 120
    while not pid_file.exists():
        if driver.poll() is not None:
            log.close()
            print(read_log(log_path)[-3000:])
            fail(f"{name}: the driver exited ({driver.returncode}) before the trainer started")
        if time.monotonic() > deadline:
            driver.kill()
            fail(f"{name}: the trainer did not start within 120 s")
        time.sleep(0.2)
    pids = json.loads(pid_file.read_text())
    tree = [pids["trainer"], pids["grandchild"]]
    for pid in tree:
        if not alive(pid):
            fail(f"{name}: pid {pid} was dead before any teardown")
    ok(f"{name}: trainer {tree[0]} and grandchild {tree[1]} are running")
    return driver, log, log_path, tree


def check_timeout(blut: Path, work: Path, dataset: Path, expect: str) -> None:
    soft = 20
    driver, log, log_path, tree = start_hanging(
        blut, work, dataset, "timeout",
        {"soft": {"secs": soft, "nanos": 0}, "hard": {"secs": soft + 120, "nanos": 0}})
    try:
        code = driver.wait(timeout=soft + 180)
    except subprocess.TimeoutExpired:
        driver.kill()
        fail("timeout: the driver did not finish after the stage timeout")
    log.close()
    text = read_log(log_path)
    print(text[-3000:])
    if code == 0:
        fail("timeout: a timed-out plan exited 0")
    if not re.search(rf"{UNIT}: cancelled; stopping its process tree \({expect}\)", text):
        fail("timeout: no tree stop was logged for the timed-out stage")
    if "timed out" not in text:
        fail("timeout: the failure does not say the stage timed out")
    ok(f"timeout: the soft timeout cancelled the stage (exit {code}) and stopped its tree")
    wait_dead(tree, "timeout")
    if expect == "systemd" and unit_active(text):
        fail(f"timeout: unit still active: {unit_active(text)}")


def check_sigterm(blut: Path, work: Path, dataset: Path, expect: str) -> None:
    driver, log, log_path, tree = start_hanging(blut, work, dataset, "sigterm", None)
    driver.send_signal(signal.SIGTERM)
    try:
        code = driver.wait(timeout=120)
    except subprocess.TimeoutExpired:
        driver.kill()
        fail("sigterm: the driver did not exit within 120 s of SIGTERM")
    log.close()
    text = read_log(log_path)
    print(text[-2000:])
    ok(f"sigterm: the driver exited ({code}) on SIGTERM")
    wait_dead(tree, "sigterm")
    if expect == "systemd" and unit_active(text):
        fail(f"sigterm: unit still active: {unit_active(text)}")


def check_no_contain(blut: Path, work: Path, dataset: Path) -> None:
    code, text = run_plan(blut, plan(work, dataset, "no_contain", None),
                          blut_env(work, BLUT_NO_CONTAIN="1"), work / "no-contain.log")
    if code != 0:
        print(text[-3000:])
        fail(f"no-contain: the plan exited {code}")
    kind, applied = placement(text)
    expected = "no memory cap, no tree teardown (BLUT_NO_CONTAIN is set)"
    if kind != "bare" or applied.strip() != expected:
        fail(f"no-contain: placement says {kind}, {applied!r}")
    line = re.search(rf"containment: {UNIT} placed by .*", text).group(0)
    ok(f"no-contain: {line}")


def check_no_bus(blut: Path, work: Path, dataset: Path) -> None:
    env = blut_env(work)
    for key in ("DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR"):
        env.pop(key, None)
    log_path = work / "no-bus.log"
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(
            [str(blut), "recipe", "declare", str(plan(work, dataset, "no_bus", None)), "--run"],
            env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300).returncode
    text = read_log(log_path)
    if code != 0:
        print(text[-3000:])
        fail(f"no-bus: the plan exited {code}")
    kind, _ = placement(text)
    if kind not in ("cgroup2", "rlimit"):
        fail(f"no-bus: placed under {kind}; expected a fallback (cgroup2 or rlimit)")
    line = re.search(rf"containment: {UNIT} placed by .*", text).group(0)
    ok(f"no-bus: {line}")


def check_no_leftover_units() -> None:
    out = subprocess.run(["systemctl", "--user", "list-units", "--all", "--plain",
                          "--no-legend", "blut-train_model-*"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        fail(f"units left behind:\n{out}")
    ok("no blut-train_model unit is left behind")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--blut", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--expect", default="systemd",
                        choices=["systemd", "cgroup2", "rlimit", "bare"])
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        fail("this check runs on Linux only")
    args.work.mkdir(parents=True, exist_ok=True)
    dataset = args.work / "train.jsonl"
    dataset.write_text("".join(json.dumps({"text": f"row {i}"}) + "\n" for i in range(8)))

    started = time.monotonic()
    check_contained(args.blut, args.work, dataset, args.expect)
    check_timeout(args.blut, args.work, dataset, args.expect)
    check_sigterm(args.blut, args.work, dataset, args.expect)
    check_no_contain(args.blut, args.work, dataset)
    check_no_bus(args.blut, args.work, dataset)
    if args.expect == "systemd":
        check_no_leftover_units()
    print(f"ALL CHECKS PASSED in {time.monotonic() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
