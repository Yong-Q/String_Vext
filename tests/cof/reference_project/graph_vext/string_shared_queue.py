"""Durable, per-task locked String queue shared by one-GPU Slurm workers."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import TextIO


@dataclass
class Claim:
    task: dict
    attempt: int
    owner: str
    handle: TextIO


class SharedQueue:
    def __init__(self, root: Path, tasks: list[dict], binding: str,
                 max_attempts: int = 2):
        self.root = Path(root)
        self.tasks = tasks
        self.binding = binding
        self.max_attempts = max_attempts
        identifiers = [task["task_id"] for task in tasks]
        if (not tasks or len(identifiers) != len(set(identifiers))
                or any(not re.fullmatch(r"[0-9a-f]{64}", task_id)
                       for task_id in identifiers)
                or max_attempts < 1):
            raise ValueError("nonempty unique SHA256 task IDs and positive retry bound required")
        identity = hashlib.sha256("\n".join(identifiers).encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in ("claims", "attempts", "done", "terminal"):
            (self.root / directory).mkdir(exist_ok=True)
        meta = self.root / "binding.json"
        expected = {"manifest_binding": binding, "task_ids_sha256": identity,
                    "tasks": len(tasks), "max_attempts": max_attempts}
        if meta.exists():
            if json.loads(meta.read_text()) != expected:
                raise ValueError("shared queue source/order/retry policy changed")
        else:
            with meta.open("x") as handle:
                json.dump(expected, handle, sort_keys=True)
        for name in ("cursor", "sweep_cursor"):
            target = self.root / name
            if not target.exists():
                with target.open("x") as handle:
                    handle.write("0\n")
        (self.root / "queue.lock").touch(exist_ok=True)

    def _mark(self, directory: str, task_id: str, payload: dict):
        path = self.root / directory / f"{task_id}.json"
        with path.open("x") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def bootstrap_completed(self, identifiers: set[str]):
        known = {task["task_id"] for task in self.tasks}
        if identifiers - known:
            raise ValueError("completed task not in shared manifest")
        with (self.root / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for task_id in identifiers:
                if not (self.root / "done" / f"{task_id}.json").exists():
                    self._mark("done", task_id, {"source": "existing_durable_csv"})

    def _acquire(self, task: dict, owner: str):
        task_id = task["task_id"]
        if (self.root / "done" / f"{task_id}.json").exists() or \
                (self.root / "terminal" / f"{task_id}.json").exists():
            return None
        handle = (self.root / "claims" / f"{task_id}.lock").open("a+")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        if (self.root / "done" / f"{task_id}.json").exists() or \
                (self.root / "terminal" / f"{task_id}.json").exists():
            self.abandon(Claim(task, 0, owner, handle))
            return None
        attempts_path = self.root / "attempts" / f"{task_id}.txt"
        attempt = int(attempts_path.read_text()) + 1 if attempts_path.exists() else 1
        if attempt > self.max_attempts:
            self._mark("terminal", task_id,
                       {"error": "retry_budget_exhausted_without_success_marker",
                        "attempts": attempt - 1})
            self.abandon(Claim(task, 0, owner, handle))
            return None
        attempts_path.write_text(f"{attempt}\n")
        handle.seek(0)
        handle.truncate()
        json.dump({"owner": owner, "attempt": attempt, "claimed_at": time.time()}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        return Claim(task, attempt, owner, handle)

    def claim_next(self, owner: str):
        with (self.root / "queue.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            cursor_path = self.root / "cursor"
            cursor = int(cursor_path.read_text())
            while cursor < len(self.tasks):
                task = self.tasks[cursor]
                cursor += 1
                cursor_path.write_text(f"{cursor}\n")
                claim = self._acquire(task, owner)
                if claim is not None:
                    return claim
            sweep_path = self.root / "sweep_cursor"
            sweep = int(sweep_path.read_text())
            for offset in range(min(512, len(self.tasks))):
                index = (sweep + offset) % len(self.tasks)
                claim = self._acquire(self.tasks[index], owner)
                if claim is not None:
                    sweep_path.write_text(f"{index + 1}\n")
                    return claim
            sweep_path.write_text(f"{(sweep + min(512, len(self.tasks))) % len(self.tasks)}\n")
            return None

    @staticmethod
    def abandon(claim: Claim):
        fcntl.flock(claim.handle, fcntl.LOCK_UN)
        claim.handle.close()

    def finish(self, claim: Claim, success: bool, error: str = ""):
        task_id = claim.task["task_id"]
        try:
            if success:
                self._mark("done", task_id, {"owner": claim.owner,
                                             "attempt": claim.attempt})
            elif claim.attempt >= self.max_attempts:
                self._mark("terminal", task_id, {"owner": claim.owner,
                                                 "attempt": claim.attempt,
                                                 "error": error})
        finally:
            self.abandon(claim)

    def is_drained(self):
        if int((self.root / "cursor").read_text()) < len(self.tasks):
            return False
        done = sum(1 for _ in (self.root / "done").glob("*.json"))
        terminal = sum(1 for _ in (self.root / "terminal").glob("*.json"))
        return done + terminal == len(self.tasks)
