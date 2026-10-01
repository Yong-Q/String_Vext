import csv
import unittest
from pathlib import Path

import numpy as np

from graph_vext.cof8948_string_runner import load_plan, cpu_bodyx_pose_energies
from graph_vext.rebuild_legacy_string_ff_vext import parse_string_definition


class CorrectedStringRunnerTests(unittest.TestCase):
    def test_full_manifest_has_all_new_guest_inputs_and_six_shards(self):
        tasks, _ = load_plan()
        self.assertEqual(len(tasks), 53688)
        self.assertEqual({int(row["shard"]) for row in tasks}, set(range(6)))
        self.assertEqual(len({row["task_id"] for row in tasks}), 53688)

    def test_bodyx_saved_pose_replay_is_finite_under_new_force_field(self):
        tasks, _ = load_plan()
        row = tasks[0]
        case = parse_string_definition(Path(row["input_path"]).read_bytes())
        self.assertTrue(np.allclose(case["guest_xyz"][:, 1:], 0))
        poses = np.loadtxt(row["saved_path"])[::100]
        energy = cpu_bodyx_pose_energies(case, poses)
        self.assertEqual(energy.shape, (5,))
        self.assertTrue(np.isfinite(energy).all())


if __name__ == "__main__":
    unittest.main()
