import unittest

from graph_vext.cof8948_vext_union import choose_source


class VextUnionTests(unittest.TestCase):
    def test_exactly_one_old_or_new_source_is_required(self):
        self.assertEqual(choose_source(True, False), "old")
        self.assertEqual(choose_source(False, True), "exact_v2")
        with self.assertRaises(ValueError):
            choose_source(True, True)
        with self.assertRaises(ValueError):
            choose_source(False, False)


if __name__ == "__main__":
    unittest.main()
