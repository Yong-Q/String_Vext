import tempfile
import unittest
from pathlib import Path

import numpy as np

from string_vext.input import parse_string_input
from string_vext.orientation import regular_pose_grid
from string_vext.physics import direct_pose_energy
from string_vext.vext import calculate_vext


SYNTHETIC_INPUT = """Nmaxa Nmaxb Nmaxc:
20 20 20
La Lb Lc dL
12.0 13.0 14.0 0.5
Alpha Beta Gamma
90.0 115.152 90.0
cutoff(A) FH_signal Mass(g/mol) Tempearture(K) Running_steps
6.0 1 30.0 298.0 100
---------String Calculation Settings---------
Direction
1
#_of_points delta_frac delta_angle_degree
401 0.0001 1.0
convergence_setting
default
------------------Adsorbate------------------
Number of sites
2
x y z Epsilon Sigma Charge Mass
-0.77 0 0 108.0 3.76 0 15.0
0.77 0 0 108.0 3.76 0 15.0
------------------Adsorbent-----------------
Number of atoms
1
ID diameter Epsilon Charge mass frac_x frac_y frac_z atom_name
1 3.4 50.0 0 12.0 0.1 0.2 0.3 C
"""


class PublicApiTests(unittest.TestCase):
    def test_input_angles_are_converted_to_triclinic_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.dat"
            path.write_text(SYNTHETIC_INPUT)
            case = parse_string_input(path)
            self.assertLess(case["cell"][2, 0], 0)
            self.assertAlmostEqual(np.linalg.norm(case["cell"][2]), 14.0)
            self.assertAlmostEqual(case["cell_angles_deg"][1], 115.152)

    def test_native_vext_matches_direct_for_synthetic_skew_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.dat"
            path.write_text(SYNTHETIC_INPUT)
            case = parse_string_input(path)
            fields = calculate_vext(case, size=3, orientations=4)
            centers, axes = regular_pose_grid(3, 4)
            direct = direct_pose_energy(case, centers, axes)
            np.testing.assert_allclose(fields["orientation_K"].reshape(4, -1).T,
                                       direct, rtol=1e-6, atol=1e-3)
            self.assertEqual(fields["site_K"].shape, (2, 3, 3, 3))


if __name__ == "__main__":
    unittest.main()
