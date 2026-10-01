import ctypes
import itertools
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'native/string_triclinic_v1/source_code'


class NativePatchExists(unittest.TestCase):
    def test_corrected_pair_header_exists(self):
        self.assertTrue((SOURCE/'triclinic_pair.h').is_file(), 'shared full-lattice pair fix missing')

    def test_output_is_final_iterate_not_unchecked_initial_path(self):
        source = (SOURCE/'RC_main.cu').read_text()
        export = source[source.index('    if (argc==3'):]
        self.assertNotIn('if (D_1>D_2)', export)
        self.assertNotIn('s0_a_ini[i]', export)
        self.assertIn('s0_a_final[i]', export)


@unittest.skipUnless((SOURCE/'triclinic_pair.h').is_file(), 'patch not yet implemented')
class NativePairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='string_pair_test_')
        output = Path(cls.directory.name)/'pair.so'
        subprocess.run(['g++', '-std=c++11', '-shared', '-fPIC', '-O2',
                        str(ROOT/'tests/native/string_triclinic_pair_host.cpp'), '-I', str(SOURCE),
                        '-o', str(output)], check=True, capture_output=True, timeout=30)
        cls.library = ctypes.CDLL(str(output))
        cls.function = cls.library.run_triclinic_pair
        cls.function.argtypes = [ctypes.POINTER(ctypes.c_double)]
        cls.function.restype = ctypes.c_double

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def evaluate(self, cell, site, atom, sigma=3.5, epsilon=98., cutoff=5.5):
        p = np.r_[site, atom, cell.T.reshape(-1), epsilon, sigma, 50., 3.4, cutoff].astype(np.float64)
        return self.function(p.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    def brute(self, cell, site, atom, sigma=3.5, epsilon=98., cutoff=5.5):
        # Independent oversized image enumeration, not the patched range logic.
        shift = np.array(list(itertools.product(range(-5, 6), repeat=3)))
        guest = ((site @ np.linalg.inv(cell)) % 1) @ cell
        images = ((atom % 1)[None]+shift) @ cell
        r = np.linalg.norm(guest-images, axis=1)
        s = (sigma+3.4)/2; e = np.sqrt(epsilon*50.)
        ratio = (s/np.maximum(r, .1*s))**6; cut = (s/cutoff)**6
        return np.where(np.maximum(r, .1*s) < cutoff, 4*e*(ratio**2-ratio-cut**2+cut), 0.).sum()

    def test_skew_cell_counterexample_includes_missed_image(self):
        h = np.array([[10., 0., 0.], [8., 6., 0.], [0., 0., 10.]])
        site = np.array([0., 0., 5.]); atom = np.array([0., .9, .5])
        self.assertAlmostEqual(self.evaluate(h, site, atom, cutoff=4), self.brute(h, site, atom, cutoff=4), places=5)
        self.assertGreater(self.evaluate(h, site, atom, cutoff=4), 1e6)

    def test_random_skew_and_orthogonal_images_match_brute(self):
        rng = np.random.default_rng(20260929)
        for h in (np.diag([9., 10., 11.]), np.array([[10., 0, 0], [4., 8., 0], [-3., 2., 9.]])):
            for _ in range(32):
                f, atom = rng.random((2, 3)); site = f @ h
                np.testing.assert_allclose(self.evaluate(h, site, atom), self.brute(h, site, atom),
                                           rtol=2e-10, atol=1e-6)

    def test_all_cutoff_images_not_only_one_nearest_image(self):
        h = np.diag([4., 4., 4.]); atom = np.array([.2, .3, .4]); site = np.array([2., 2., 2.])
        np.testing.assert_allclose(self.evaluate(h, site, atom, cutoff=6),
                                   self.brute(h, site, atom, cutoff=6), rtol=2e-10)

    def test_unwrapped_guest_and_atom_are_periodic(self):
        h = np.array([[10., 0, 0], [4., 8., 0], [-3., 2., 9.]])
        f, atom = np.array([.25, .35, .6]), np.array([.8, .1, .2])
        value = self.evaluate(h, f @ h, atom)
        for shift in (np.array([3, -2, 1]), np.array([-5, 1, -4])):
            np.testing.assert_allclose(value, self.evaluate(h, (f+shift) @ h, atom), rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(value, self.evaluate(h, f @ h, atom+shift), rtol=1e-10, atol=1e-7)

    def test_contact_floor_and_cutoff_are_finite_without_energy_cap(self):
        h = np.diag([20., 20., 20.]); atom = np.array([.5, .5, .5]); site = atom @ h
        energy = self.evaluate(h, site, atom)
        self.assertTrue(np.isfinite(energy)); self.assertGreater(energy, 1e12)
        self.assertAlmostEqual(self.evaluate(h, site+[5.5, 0, 0], atom), 0., places=10)

    def test_every_energy_route_uses_the_shared_fix(self):
        header = (SOURCE/'global.h').read_text()
        self.assertEqual(header.count('triclinic_pair_lj('), 4)
        self.assertNotIn('>  0.5*cart_z_extended_device', header)
        self.assertNotIn('> 0.5*cart_z_extended_device', header)
