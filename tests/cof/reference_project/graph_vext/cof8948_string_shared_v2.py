"""Six independent GPU dispatchers claiming from one durable String task pool."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from graph_vext import legacy8238_runner_v7 as control
from graph_vext import cof8948_string_runner as original
from graph_vext.string_shared_queue import SharedQueue


BASE = original.BASE
EXE = original.EXE
RESERVE_MIB = 1024
POOL = BASE / "shared_pool_v2"
PILOT = BASE / "shared_pilot_v2"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_binding():
    root = original.ROOT / "graph_vext"
    names = ("cof8948_string_shared_v2.py", "string_shared_queue.py",
             "cof8948_string_runner.py", "legacy8238_runner_v7.py")
    return {name: sha(root / name) for name in names}


def task_order(tasks):
    return sorted(tasks, key=lambda row: row["task_id"])


def _v1_gate(exe, manifest_sha):
    path = BASE / "pilot_v1/gate.json"
    gate = json.loads(path.read_text())
    binding = control.native_binding(exe)
    if (not gate.get("passed") or gate.get("manifest_sha256") != manifest_sha
            or gate.get("native_binding") != binding
            or gate.get("reserve_MiB") != RESERVE_MIB):
        raise ValueError("completed corrected BODY-X/1-GiB String pilot required")
    return gate, binding


def pilot(exe: Path = EXE):
    if PILOT.exists():
        raise FileExistsError(PILOT)
    tasks, manifest_sha = original.load_plan()
    gate, native = _v1_gate(exe, manifest_sha)
    codes = code_binding()
    ordered = task_order(tasks)
    # A direction distinct from the original pilot; no full-pool state is touched.
    picked = next(row for row in ordered if row["direction"] == "2"
                  and int(row["framework_atoms"]) < 800)
    PILOT.mkdir(parents=True)
    queue = SharedQueue(PILOT / "queue", [picked], manifest_sha)
    claim = queue.claim_next("pilot")
    if claim is None or queue.claim_next("second_worker") is not None:
        raise RuntimeError("one-task shared-pool exclusivity failed")
    uuid = control.allocated_uuid()
    before = control.query(uuid)
    budget = control.source_allocation_budget(int(picked["framework_atoms"]))
    if before["free_MiB"] < RESERVE_MIB + budget:
        queue.abandon(claim)
        raise RuntimeError("pilot GPU cannot preserve 1-GiB reserve")
    tracked = {picked["task_id"]: {"proc": None, "abort": None}}
    lock = threading.Lock()
    samples = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(original.execute, picked, PILOT, exe, uuid, tracked, lock, True)
        while not future.done():
            card = control.query(uuid)
            samples.append(card)
            if card["free_MiB"] < RESERVE_MIB:
                with lock:
                    tracked[picked["task_id"]]["abort"] = "shared pilot GPU reserve breached"
            time.sleep(.2)
        result = future.result()
    success = result["status"] in ("completed", "completed_with_review")
    queue.finish(claim, success, result.get("error", ""))
    if not success or not queue.is_drained():
        raise RuntimeError("shared String pilot failed: " + result.get("error", ""))
    if control.native_binding(exe) != native or code_binding() != codes:
        raise RuntimeError("native/runner changed during shared pilot")
    peak = max((row["used_MiB"] for row in samples), default=before["used_MiB"]) - before["used_MiB"]
    if peak > budget:
        raise RuntimeError("shared pilot measured GPU peak above source allowance")
    record = {"passed": True, "manifest_sha256": manifest_sha,
              "native_binding": native, "code_binding": codes,
              "original_pilot_sha256": sha(BASE / "pilot_v1/gate.json"),
              "reserve_MiB": RESERVE_MIB, "sampled_increment_MiB": peak,
              "budget_MiB": budget, "result": result,
              "queue_contract": "one locked claim; second worker blocked; finished marker durable"}
    control.atomic_json(PILOT / "gate.json", record)
    print(json.dumps({"passed": True, "task_id": picked["task_id"],
                      "gpu_memory_increment_MiB": peak}), flush=True)


def run(slot: int, exe: Path = EXE, maximum: int = 25, stagger: float = 5.):
    if slot not in range(6) or not 1 <= maximum <= 25 or not 3 <= stagger <= 8:
        raise ValueError("six one-GPU workers, 1-25 processes and 3-8 s stagger required")
    tasks, manifest_sha = original.load_plan()
    old_gate, native = _v1_gate(exe, manifest_sha)
    gate = json.loads((PILOT / "gate.json").read_text())
    codes = code_binding()
    if (not gate.get("passed") or gate.get("manifest_sha256") != manifest_sha
            or gate.get("native_binding") != native or gate.get("code_binding") != codes
            or gate.get("original_pilot_sha256") != sha(BASE / "pilot_v1/gate.json")
            or gate.get("reserve_MiB") != RESERVE_MIB):
        raise ValueError("matching shared-queue GPU pilot required")
    job = os.environ["SLURM_JOB_ID"]
    uuid = control.allocated_uuid()
    out = BASE / "workers_shared_v2" / control.monitor_name(job, slot, uuid)
    out.mkdir(parents=True, exist_ok=False)
    ordered = task_order(tasks)
    queue = SharedQueue(POOL, ordered, manifest_sha)
    queue.bootstrap_completed(set(control.read_completed(BASE)))
    control.atomic_json(out / "context.json", {
        "slot": slot, "job_id": job, "gpu_uuid": uuid,
        "manifest_sha256": manifest_sha, "native_binding": native,
        "code_binding": codes, "reserve_MiB": RESERVE_MIB,
        "maximum": maximum, "stagger_seconds": stagger,
        "task_pool": str(POOL), "task_count": len(ordered)})
    host_budget = int(os.environ.get("SLURM_MEM_PER_NODE", "32768"))
    tracked, active = {}, {}
    lock = threading.Lock()
    pending_claim = None
    last_launch = -float("inf")
    last_sync = time.monotonic()
    last_drained_check = -float("inf")
    completed_here = failed_here = 0
    stopping = False
    drained = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        with lock:
            for slot_state in tracked.values():
                slot_state["abort"] = "job_stop_requested"

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (out / "gpu_memory.csv").open("x") as memory, \
            ThreadPoolExecutor(max_workers=maximum) as pool:
        memory.write("time,gpu_uuid,used_MiB,free_MiB,active,pending_MiB,next_gap_seconds\n")
        monitor_errors = 0
        while True:
            for future, claim in list(active.items()):
                if not future.done():
                    continue
                key = claim.task["task_id"]
                try:
                    result = future.result()
                    control.publish(BASE, result)
                    success = result["status"] in ("completed", "completed_with_review")
                    queue.finish(claim, success, result.get("error", ""))
                    completed_here += int(success)
                    failed_here += int(not success)
                except Exception as exc:
                    queue.finish(claim, False, repr(exc))
                    failed_here += 1
                del active[future]
                with lock:
                    tracked.pop(key, None)
            if stopping and not active:
                break
            try:
                card = control.query(uuid)
                allocated = control.process_memory(uuid)
                monitor_errors = 0
                with lock:
                    for key, slot_state in tracked.items():
                        proc = slot_state["proc"]
                        pid = proc.pid if proc else None
                        control.update_allocation_phase(slot_state,
                            Path(slot_state["stderr_path"]), allocated.get(pid, 0.), control.rss(pid))
                    snapshots = [dict(value) for value in tracked.values()]
                    if card["free_MiB"] < RESERVE_MIB:
                        stopping = True
                        if snapshots:
                            tracked[next(reversed(tracked))]["abort"] = "GPU reserve breached"
                pending_memory = sum(max(0., item["estimate"] - item["allocated"])
                                     for item in snapshots)
                gap = control.launch_interval(card["used_MiB"], stagger)
                memory.write(f"{time.time()},{uuid},{card['used_MiB']},{card['free_MiB']},"
                             f"{len(active)},{pending_memory},{gap}\n")
                memory.flush()
                control.atomic_json(out / "gpu_latest.json", {
                    **card, "slot": slot, "active": len(active),
                    "pending_MiB": pending_memory, "completed_here": completed_here,
                    "failed_here": failed_here, "next_gap_seconds": gap})
                if not stopping and len(active) < maximum and gap is not None \
                        and time.monotonic() - last_launch >= gap:
                    if pending_claim is None:
                        pending_claim = queue.claim_next(f"job_{job}_slot_{slot}_{uuid}")
                    if pending_claim is not None:
                        task = pending_claim.task
                        if pending_claim.attempt > 1 and \
                                task["task_id"] in control.read_completed(BASE):
                            queue.finish(pending_claim, True)
                            pending_claim = None
                            continue
                        atoms = int(task["framework_atoms"])
                        budget = control.calibrated_start_budget(atoms, old_gate["profiles"])
                        if control.admit(card["free_MiB"], snapshots, budget,
                                         RESERVE_MIB, maximum, host_budget):
                            key = task["task_id"]
                            attempt_root = out / "attempts" / f"try{pending_claim.attempt}"
                            attempt_root.mkdir(parents=True, exist_ok=True)
                            slot_state = {"estimate": budget, "allocated": 0.,
                                          "rss": 0., "atoms": atoms,
                                          "phase": "initial", "host_estimate": budget * 1.5,
                                          "proc": None, "abort": None,
                                          "stderr_path": str(attempt_root / key / "native.stderr")}
                            with lock:
                                tracked[key] = slot_state
                            active[pool.submit(original.execute, task, attempt_root, exe, uuid,
                                               tracked, lock)] = pending_claim
                            pending_claim = None
                            last_launch = time.monotonic()
                if pending_claim is None and not active and \
                        time.monotonic() - last_drained_check > 30:
                    last_drained_check = time.monotonic()
                    if time.monotonic() - last_sync > 60:
                        queue.bootstrap_completed(set(control.read_completed(BASE)))
                        last_sync = time.monotonic()
                    drained = queue.is_drained()
                    if drained:
                        break
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                monitor_errors += 1
                control.atomic_json(out / "monitor_failure.json", {
                    "error": str(exc), "consecutive_failures": monitor_errors})
                if monitor_errors >= 3:
                    stop(None, None)
            time.sleep(2 if active or pending_claim is not None else 10)
    if pending_claim is not None:
        queue.abandon(pending_claim)
    if code_binding() != codes or control.native_binding(exe) != native:
        raise RuntimeError("String source or binary changed during shared run")
    terminal = sum(1 for _ in (POOL / "terminal").glob("*.json"))
    control.atomic_json(out / "report.json", {
        "slot": slot, "job_id": job, "completed_here": completed_here,
        "failed_attempts_here": failed_here, "drained": drained,
        "terminal_tasks": terminal, "stopped": stopping,
        "manifest_sha256": manifest_sha, "code_binding": codes})
    if stopping or not drained or terminal:
        raise RuntimeError("shared pool incomplete; durable successes retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "run"), required=True)
    parser.add_argument("--slot", type=int)
    parser.add_argument("--max-concurrent", type=int, default=25)
    parser.add_argument("--stagger", type=float, default=5.)
    args = parser.parse_args()
    if args.mode == "pilot":
        pilot()
    else:
        if args.slot is None:
            parser.error("run mode requires --slot")
        run(args.slot, maximum=args.max_concurrent, stagger=args.stagger)


if __name__ == "__main__":
    main()
