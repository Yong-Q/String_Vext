import csv
import tempfile
import unittest
from pathlib import Path

from graph_vext.cof8948_oldff_materialize import materialize
from graph_vext.cof8948_oldff_prepare import prepare
from graph_vext.rebuild_legacy_string_ff_vext import parse_string_definition


ROOT = Path(__file__).resolve().parents[1]


class MaterializeTests(unittest.TestCase):
    def test_six_oldff_inputs_are_materialized_without_changing_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            prepared = Path(tmp) / "prepared"
            prepare(prepared, limit=1)
            with (prepared / "tasks.csv").open(newline="") as handle:
                tasks = list(csv.DictReader(handle))
            result = materialize(prepared, workers=2)
            self.assertEqual(result["inputs"], 6)
            self.assertEqual(result["created"], 6)
            gas = next(task for task in tasks if task["gas"] == "C2H6")
            original = parse_string_definition(Path(gas["source_input_path"]).read_bytes())
            target = (prepared / "inputs" / gas["gas"] / gas["name"] /
                      f"dir{gas['direction']}" / "input.dat")
            converted = parse_string_definition(target.read_bytes())
            self.assertEqual(float(original["guest_epsilon"][0]), 98.0)
            self.assertEqual(float(converted["guest_epsilon"][0]), 108.0)
            again = materialize(prepared, workers=2)
            self.assertEqual(again["inputs"], 6)

    def test_changed_existing_input_is_never_overwritten(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "runs") as tmp:
            prepared = Path(tmp) / "prepared"
            prepare(prepared, limit=1)
            materialize(prepared, workers=2)
            with (prepared / "tasks.csv").open(newline="") as handle:
                task = next(csv.DictReader(handle))
            target = (prepared / "inputs" / task["gas"] / task["name"] /
                      f"dir{task['direction']}" / "input.dat")
            target.write_text("tampered evidence")
            with self.assertRaisesRegex(ValueError, "existing input hash"):
                materialize(prepared, workers=2)
            self.assertEqual(target.read_text(), "tampered evidence")


if __name__ == "__main__":
    unittest.main()
