import csv
import unittest

import numpy as np

from graph_vext.cof8948_oldff_vext import PREPARED, verify_task
from graph_vext.exact_neighbor_reuse_probe import ReusePotential
from graph_vext.string_ff_vext import direct_pose_energy


class NeighborReuseTests(unittest.TestCase):
    def test_both_gases_match_direct_in_skew_cell_and_at_boundaries(self):
        with (PREPARED / "tasks.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        centers = np.array([[0.001, 0.999, 0.5], [0.25, 0.5, 0.75],
                            [0.98, 0.03, 0.02], [0.5, 0.5, 0.5]])
        axes = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [0.5, 0.5, np.sqrt(0.5)]])
        for gas in ("C2H4", "C2H6"):
            row = next(row for row in rows if row["gas"] == gas
                       and row["direction"] == "1")
            case = verify_task(PREPARED, row)
            reference = direct_pose_energy(case, centers, axes)
            actual = ReusePotential(case).evaluate(centers, axes)
            np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
