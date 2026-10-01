import csv
import unittest

import numpy as np

from graph_vext.cof8948_oldff_vext import PREPARED, compute_one_field, verify_task
from graph_vext.cof8948_oldff_vext_exact import compute_one_field_exact


class ExactVextTests(unittest.TestCase):
    def test_exact_neighbor_reuse_matches_all_direct_field_channels(self):
        with (PREPARED / "tasks.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for gas in ("C2H4", "C2H6"):
            row = next(row for row in rows if row["gas"] == gas
                       and row["direction"] == "1")
            case = verify_task(PREPARED, row)
            reference = compute_one_field(case, size=4, orientations=8)
            candidate = compute_one_field_exact(case, size=4, orientations=8)
            self.assertEqual(set(candidate), set(reference))
            for key in reference:
                np.testing.assert_allclose(candidate[key], reference[key],
                                           rtol=1e-6, atol=1e-3,
                                           err_msg=f"{gas} {key}")


if __name__ == "__main__":
    unittest.main()
