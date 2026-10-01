import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from graph_vext.cof8948_oldff_vext import PREPARED, verify_task
from graph_vext.exact_neighbor_reuse_native import NativeReusePotential
from graph_vext.string_ff_vext import direct_pose_energy


class NativeNeighborReuseTests(unittest.TestCase):
    def test_native_kernel_matches_direct_across_skew_boundaries(self):
        source = Path(__file__).resolve().parents[1] / "native/vext_exact_reuse_v1/kernel.cpp"
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "kernel.so"
            subprocess.run(["g++", "-O3", "-std=c++17", "-fPIC", "-shared", "-nostdlib",
                            str(source), "-o", str(library)], check=True)
            with (PREPARED / "tasks.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            centers = np.array([[.001, .999, .5], [.999, .001, .5],
                                [.25, .5, .75], [.5, .5, .5]])
            axes = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                             [1., 1., 1.]])
            for gas in ("C2H4", "C2H6"):
                row = next(row for row in rows if row["gas"] == gas
                           and row["direction"] == "1")
                case = verify_task(PREPARED, row)
                reference = direct_pose_energy(case, centers, axes)
                actual = NativeReusePotential(case, library).evaluate(centers, axes)
                np.testing.assert_allclose(actual, reference, rtol=1e-10, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
