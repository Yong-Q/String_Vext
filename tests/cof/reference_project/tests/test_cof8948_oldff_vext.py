import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_vext import (
    verify_task, compute_one_field, load_plan, build_one, code_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARED = ROOT / "runs/cof8948_string_oldff_v1/prepared_smoke100"


class CorrectedVextTests(unittest.TestCase):
    def _task(self, gas):
        with (PREPARED / "tasks.csv").open(newline="") as handle:
            return next(row for row in csv.DictReader(handle)
                        if row["gas"] == gas and row["direction"] == "1")

    def test_new_inputs_pass_and_original_98K_inputs_fail(self):
        for gas, expected_epsilon in (("C2H4", 92.8), ("C2H6", 108.0)):
            row = self._task(gas)
            case = verify_task(PREPARED, row)
            self.assertAlmostEqual(float(case["guest_epsilon"][0]), expected_epsilon)
            self.assertFalse(np.allclose(case["cell"][2], [0, 0, case["cell"][2, 2]]))
            with self.assertRaisesRegex(ValueError, "input hash|guest"):
                verify_task(PREPARED, row, override=Path(row["source_input_path"]))

    def test_small_direct_field_is_finite(self):
        case = verify_task(PREPARED, self._task("C2H6"))
        result = compute_one_field(case, size=3, orientations=4)
        self.assertEqual(result["orientation_K"].shape, (4, 3, 3, 3))
        self.assertTrue(np.isfinite(result["grid"]).all())

    def test_versioned_material_output_resumes_without_overwrite(self):
        tasks, binding = load_plan(PREPARED, expected_materials=100)
        name = tasks[0]["name"]
        selected = [task for task in tasks if task["name"] == name]
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            output = Path(tmp)
            first = build_one(name, selected, PREPARED, output, 3, 4,
                              code_hashes(), binding, resume=False)
            self.assertEqual(first["name"], name)
            self.assertEqual(first["gases"]["C2H6"]["guest_epsilon_K"], 108.0)
            cached = build_one(name, selected, PREPARED, output, 3, 4,
                               code_hashes(), binding, resume=True)
            self.assertEqual(cached["sha256"], first["sha256"])
            with self.assertRaises(FileExistsError):
                build_one(name, selected, PREPARED, output, 3, 4,
                          code_hashes(), binding, resume=False)


if __name__ == "__main__":
    unittest.main()
