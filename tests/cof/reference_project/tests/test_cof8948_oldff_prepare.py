import tempfile
import unittest
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_prepare import prepare, rewrite_input, verify_cif_alignment
from graph_vext.rebuild_legacy_string_ff_vext import parse_string_definition


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/home/qiuyong/gcmc_agent/pormake_output/active_learning_cof30k_diffusion_test"
)
NAME = "bne+C17+C32+L11_CF3"


class OldForceFieldPreparationTests(unittest.TestCase):
    def _original(self, gas):
        return (SOURCE / f"string_runs_{gas}" / "job_231" / NAME /
                "dir1" / "input.dat").read_bytes()

    def test_rewrite_changes_only_two_site_guest_and_mass(self):
        for gas, sigma, epsilon, bond, mass in (
            ("C2H4", 3.68, 92.8, 1.33, 28.0),
            ("C2H6", 3.76, 108.0, 1.54, 30.0),
        ):
            with self.subTest(gas=gas):
                original = self._original(gas)
                converted = rewrite_input(original, gas, 1)
                before = parse_string_definition(original)
                after = parse_string_definition(converted)
                for field in ("cell", "frame_frac", "frame_sigma", "frame_epsilon",
                              "frame_mass"):
                    np.testing.assert_array_equal(before[field], after[field])
                self.assertAlmostEqual(float(after["guest_sigma"][0]), sigma)
                self.assertAlmostEqual(float(after["guest_epsilon"][0]), epsilon)
                self.assertAlmostEqual(float(after["total_mass"]), mass)
                self.assertAlmostEqual(float(after["original_body"][1, 0] -
                                             after["original_body"][0, 0]), bond)
                self.assertAlmostEqual(float(after["cell"][2, 0]),
                                       float(before["cell"][2, 0]))

    def test_unexpected_old_site_parameters_are_rejected(self):
        original = self._original("C2H6")
        tampered = original.replace(b"98.000000 3.750000", b"99.000000 3.750000", 1)
        self.assertNotEqual(tampered, original)
        with self.assertRaisesRegex(ValueError, "unexpected source guest"):
            rewrite_input(tampered, "C2H6", 1)

    def test_nonorthogonal_cif_and_atom_order_match_new_input(self):
        cif = ROOT / "inputs/selected_cifs" / f"{NAME}.cif"
        converted = rewrite_input(self._original("C2H6"), "C2H6", 1)
        report = verify_cif_alignment(cif, converted)
        self.assertGreater(report["max_angle_deviation_deg"], 20)
        self.assertLess(report["max_periodic_atom_error_A"], 2e-4)

    def test_small_manifest_has_six_directions_per_real_cof(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as temporary:
            output = Path(temporary) / "prepared"
            report = prepare(output, limit=2)
            self.assertEqual(report["materials"], 2)
            self.assertEqual(report["tasks"], 12)
            self.assertTrue(report["passed"])
            self.assertTrue((output / "tasks.csv").is_file())
            with self.assertRaises(FileExistsError):
                prepare(output, limit=2)


if __name__ == "__main__":
    unittest.main()
