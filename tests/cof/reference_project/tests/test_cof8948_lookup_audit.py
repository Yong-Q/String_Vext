import unittest

import numpy as np

from graph_vext.cof8948_lookup_audit import lookup_poses, summarize_error


class LookupAuditTests(unittest.TestCase):
    def test_constant_site_field_is_orientation_independent_in_skew_cell(self):
        cell = np.array([[12., 0., 0.], [3., 11., 0.], [1., 2., 10.]])
        site = np.full((2, 4, 4, 4), 7., dtype=np.float32)
        centers = np.array([[.125, .375, .625], [.875, .875, .125]])
        axes = np.array([[1., 0., 0.], [0., 1., 0.]])
        result = lookup_poses(site, cell, centers, axes, 1.54)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result, 14., atol=1e-6)

    def test_error_summary_counts_only_accessible_poses(self):
        reference = np.array([[0., 100., 5000.], [10., -20., 6000.]])
        estimate = np.array([[0., 109., 10000.], [20., -31., 10000.]])
        summary = summarize_error(reference, estimate, reference < 2000.)
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["within_10K"], 3)
        self.assertAlmostEqual(summary["mae_K"], 7.5)
        self.assertAlmostEqual(summary["max_K"], 11.)


if __name__ == "__main__":
    unittest.main()
