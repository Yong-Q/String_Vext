import unittest


class TwoGiBGuardTests(unittest.TestCase):
    def test_peak_admission_retains_two_gib(self):
        from graph_vext.legacy8238_runner_v7 import admit
        self.assertFalse(admit(5000,[],3200,2048,25,32768))
        self.assertTrue(admit(5300,[],3200,2048,25,32768))

    def test_requested_three_five_eight_second_schedule(self):
        from graph_vext.legacy8238_runner_v7 import launch_interval,launch_count
        self.assertEqual(launch_interval(14000),3)
        self.assertEqual(launch_count(14000),2)
        self.assertEqual(launch_interval(15360),5)
        self.assertEqual(launch_interval(18432),8)
        self.assertEqual(launch_count(18432),1)
        self.assertEqual(launch_interval(20500),8)
        self.assertEqual(launch_count(20500),1)
        self.assertIsNone(launch_interval(22528))
        self.assertEqual(launch_count(22528),0)

    def test_pilot_records_two_gib_reserve(self):
        from inspect import getsource
        from graph_vext.legacy8238_runner_v7 import pilot,run
        self.assertIn('reserve=2048',getsource(pilot))
        self.assertIn('reserve=2048',getsource(run))
