import tempfile
import unittest
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from string_vext.shared_queue import SharedQueue


def claim_until_drained(root, tasks, owner):
    queue = SharedQueue(Path(root), tasks, "binding")
    claimed = []
    while True:
        claim = queue.claim_next(owner)
        if claim is None:
            if queue.is_drained():
                return claimed
            continue
        claimed.append(claim.task["task_id"])
        queue.finish(claim, success=True)


class SharedQueueTests(unittest.TestCase):
    def _tasks(self, n):
        return [{"task_id": f"{i:064x}", "name": f"material_{i}"} for i in range(n)]

    def test_two_workers_claim_distinct_tasks_and_retry_failed_one(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = self._tasks(3)
            first = SharedQueue(Path(directory), tasks, "binding")
            second = SharedQueue(Path(directory), tasks, "binding")
            a = first.claim_next("worker_a")
            b = second.claim_next("worker_b")
            self.assertNotEqual(a.task["task_id"], b.task["task_id"])
            first.finish(a, success=True)
            second.finish(b, success=False, error="not converged")
            c = first.claim_next("worker_a")
            self.assertEqual(c.task["task_id"], tasks[2]["task_id"])
            first.finish(c, success=True)
            retry = second.claim_next("worker_b")
            self.assertEqual(retry.task["task_id"], b.task["task_id"])
            self.assertEqual(retry.attempt, 2)
            second.finish(retry, success=False, error="not converged")
            self.assertTrue(first.is_drained())
            self.assertIsNone(first.claim_next("worker_a"))

    def test_abandoned_claim_can_be_recovered_after_lock_release(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = self._tasks(1)
            first = SharedQueue(Path(directory), tasks, "binding")
            second = SharedQueue(Path(directory), tasks, "binding")
            a = first.claim_next("worker_a")
            self.assertIsNone(second.claim_next("worker_b"))
            first.abandon(a)
            recovered = second.claim_next("worker_b")
            self.assertEqual(recovered.task["task_id"], tasks[0]["task_id"])
            second.finish(recovered, success=True)
            self.assertTrue(first.is_drained())

    def test_four_processes_share_one_hundred_tasks_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            tasks = self._tasks(100)
            queue = SharedQueue(Path(directory), tasks, "binding")
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(claim_until_drained, directory, tasks, f"worker_{i}")
                           for i in range(4)]
                observed = [task_id for future in futures for task_id in future.result(timeout=30)]
            self.assertEqual(len(observed), len(set(observed)))
            self.assertEqual(set(observed), {task["task_id"] for task in tasks})
            self.assertTrue(queue.is_drained())


if __name__ == "__main__":
    unittest.main()
