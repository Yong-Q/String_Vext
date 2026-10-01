"""Seal new-guest-FF String inputs for the distinct 8,948 Pormake COFs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from graph_vext.cif import parse_cif
from graph_vext.rebuild_legacy_string_ff_vext import parse_string_definition


ROOT = Path(__file__).resolve().parents[1]
COHORT = "cof8948_oldff_v1"
GASES = {
    "C2H4": dict(source_sigma=3.70, source_epsilon=85.0, source_bond=1.34,
                source_mass=28.054, source_site_mass=14.027,
                sigma=3.68, epsilon=92.8, bond=1.33, mass=28.0, site_mass=14.0),
    "C2H6": dict(source_sigma=3.75, source_epsilon=98.0, source_bond=1.54,
                source_mass=30.070, source_site_mass=15.035,
                sigma=3.76, epsilon=108.0, bond=1.54, mass=30.0, site_mass=15.0),
}
FIELDS = ("task_id", "task_index", "material_index", "name", "gas", "direction",
          "shard", "cif_path", "cif_sha256", "framework_atoms", "source_input_path",
          "source_input_sha256", "new_input_sha256", "saved_path",
          "saved_path_sha256")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _after(lines: list[str], marker: str) -> int:
    found = [i for i, value in enumerate(lines) if value.strip().startswith(marker)]
    if len(found) != 1:
        raise ValueError(f"expected exactly one {marker!r} header")
    return found[0] + 1


def rewrite_input(original: bytes, gas: str, direction: int) -> bytes:
    """Change only guest-site definition and total guest mass; keep COF FF."""
    if gas not in GASES or direction not in (1, 2, 3):
        raise ValueError("unsupported gas or lattice direction")
    recipe = GASES[gas]
    lines = original.decode("ascii").splitlines(keepends=True)
    bare = [line.rstrip("\r\n") for line in lines]
    source = parse_string_definition(original)
    expected_xyz = np.array([[-recipe["source_bond"] / 2, 0, 0],
                             [recipe["source_bond"] / 2, 0, 0]])
    if (not np.allclose(source["original_body"], expected_xyz, rtol=0, atol=1e-6)
            or not np.allclose(source["guest_sigma"], recipe["source_sigma"], rtol=0, atol=1e-6)
            or not np.allclose(source["guest_epsilon"], recipe["source_epsilon"], rtol=0, atol=1e-6)
            or not np.allclose(source["guest_mass"], recipe["source_site_mass"], rtol=0, atol=1e-6)
            or not np.isclose(source["total_mass"], recipe["source_mass"], rtol=0, atol=1e-6)):
        raise ValueError(f"unexpected source guest for {gas}")
    if (int(bare[_after(bare, "Direction")]) != direction
            or int(bare[_after(bare, "Number of sites")]) != 2
            or [float(x) for x in bare[_after(bare, "#_of_points")].split()] !=
            [401., .0001, 1.]
            or not np.allclose(source["numeric_conditions"],
                               [12.9, 1, recipe["source_mass"], 298, 10000],
                               rtol=0, atol=1e-6)):
        raise ValueError("unexpected String direction or numerical settings")
    condition = _after(bare, "cutoff(A)")
    site = _after(bare, "Number of sites") + 2
    newline = "\r\n" if b"\r\n" in original else "\n"
    values = bare[condition].split()
    values[2] = f"{recipe['mass']:.6f}"
    lines[condition] = " ".join(values) + newline
    for index, sign in enumerate((-1, 1)):
        lines[site + index] = (
            f"{sign * recipe['bond'] / 2:.6f} 0.000000 0.000000 "
            f"{recipe['epsilon']:.6f} {recipe['sigma']:.6f} 0.000000 "
            f"{recipe['site_mass']:.6f}{newline}"
        )
    converted = "".join(lines).encode("ascii")
    target = parse_string_definition(converted)
    for key in ("cell", "frame_frac", "frame_sigma", "frame_epsilon", "frame_mass"):
        if not np.array_equal(source[key], target[key]):
            raise ValueError(f"framework or CIF geometry changed: {key}")
    if source["elements"] != target["elements"]:
        raise ValueError("framework atom order changed")
    expected_new = np.array([[-recipe["bond"] / 2, 0, 0],
                             [recipe["bond"] / 2, 0, 0]])
    if (not np.allclose(target["original_body"], expected_new, atol=1e-6, rtol=0)
            or not np.allclose(target["guest_sigma"], recipe["sigma"], atol=1e-6, rtol=0)
            or not np.allclose(target["guest_epsilon"], recipe["epsilon"], atol=1e-6, rtol=0)
            or not np.allclose(target["guest_mass"], recipe["site_mass"], atol=1e-6, rtol=0)
            or not np.isclose(target["total_mass"], recipe["mass"], atol=1e-6, rtol=0)):
        raise ValueError("transformed guest definition mismatch")
    return converted


def verify_cif_alignment(cif: Path, transformed: bytes) -> dict:
    """Check full triclinic metric and CIF atom order before scheduling GPU."""
    cell, atoms = parse_cif(Path(cif))
    case = parse_string_definition(transformed)
    if len(atoms) != len(case["frame_frac"]) or not np.allclose(
        case["cell"], cell.matrix, rtol=0, atol=3e-5
    ):
        raise ValueError(f"CIF/String nonorthogonal cell or atom count mismatch: {cif}")
    if any(atom.charge != 0 for atom in atoms):
        raise ValueError(f"charged framework requires separate validation: {cif}")
    elements = [atom.element for atom in atoms]
    if case["elements"] != elements:
        raise ValueError(f"CIF/String element row order mismatch: {cif}")
    coordinates = np.array([[atom.fract_x, atom.fract_y, atom.fract_z]
                            for atom in atoms])
    delta = case["frame_frac"] - coordinates
    residual = np.linalg.norm((delta - np.rint(delta)) @ case["cell"], axis=1)
    maximum = float(residual.max())
    if maximum > 2e-4:
        raise ValueError(f"CIF/String periodic atom geometry mismatch: {cif} {maximum:.4g}A")
    return dict(atoms=len(atoms), max_periodic_atom_error_A=maximum,
                max_angle_deviation_deg=float(max(abs(value - 90)
                    for value in (cell.alpha, cell.beta, cell.gamma))))


def prepare(output: Path, limit: int = 0) -> dict:
    output = Path(output).resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    manifest = ROOT / "runs/v2/manifest.csv"
    source_path = ROOT / "inputs/full_pose_v4_v1/sources.json"
    names = [row["name"] for row in read_csv(manifest)]
    if len(names) != len(set(names)) or len(names) != 8948:
        raise ValueError("8,948-name COF release required")
    if limit:
        names = names[:limit]
    sources = json.loads(source_path.read_text())
    tasks = []
    max_atom_error = 0.0
    max_angle_deviation = 0.0
    for material_index, name in enumerate(names):
        cif = ROOT / "inputs/selected_cifs" / f"{name}.cif"
        cif_hash = digest(cif.read_bytes())
        listed = {(row["gas"], int(row["direction"])): row for row in sources[name]}
        expected = {(gas, axis) for gas in GASES for axis in (1, 2, 3)}
        if set(listed) != expected or len(sources[name]) != 6:
            raise ValueError(f"incomplete/duplicate source paths: {name}")
        framework = None
        checked_geometry = False
        for gas in GASES:
            for direction in (1, 2, 3):
                index = len(tasks)
                source = Path(listed[gas, direction]["source"]) / "input.dat"
                raw = source.read_bytes()
                new = rewrite_input(raw, gas, direction)
                case = parse_string_definition(new)
                if not checked_geometry:
                    geometry = verify_cif_alignment(cif, new)
                    max_atom_error = max(max_atom_error,
                                         geometry["max_periodic_atom_error_A"])
                    max_angle_deviation = max(max_angle_deviation,
                                              geometry["max_angle_deviation_deg"])
                    checked_geometry = True
                definition = tuple(case[key].tobytes() for key in
                                   ("cell", "frame_frac", "frame_sigma", "frame_epsilon", "frame_mass"))
                if framework is None:
                    framework = definition
                elif framework != definition:
                    raise ValueError(f"gas/direction framework mismatch: {name} {gas} {direction}")
                saved = Path(listed[gas, direction]["source"]) / "string_path.dat"
                if digest(saved.read_bytes()) != listed[gas, direction]["path_sha256"]:
                    raise ValueError(f"source path changed: {name} {gas} {direction}")
                tasks.append(dict(
                    task_id=digest(f"{COHORT}\0{name}\0{gas}\0{direction}".encode()),
                    task_index=index, material_index=material_index, name=name, gas=gas,
                    direction=direction, shard=material_index % 6,
                    cif_path=str(cif), cif_sha256=cif_hash,
                    framework_atoms=len(case["frame_frac"]),
                    source_input_path=str(source), source_input_sha256=digest(raw),
                    new_input_sha256=digest(new), saved_path=str(saved),
                    saved_path_sha256=listed[gas, direction]["path_sha256"],
                ))
    if len(tasks) != len(names) * 6 or len({row["task_id"] for row in tasks}) != len(tasks):
        raise ValueError("task count or IDs inconsistent")
    output.mkdir(parents=True)
    temporary = output / "tasks.tmp.csv"
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(tasks)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output / "tasks.csv")
    report = dict(passed=True, cohort=COHORT, materials=len(names), tasks=len(tasks),
                  max_periodic_atom_error_A=max_atom_error,
                  max_angle_deviation_deg=max_angle_deviation,
                  task_csv_sha256=digest((output / "tasks.csv").read_bytes()),
                  source_manifest_sha256=digest(manifest.read_bytes()),
                  source_directory_sha256=digest(source_path.read_bytes()),
                  transformer_sha256=digest(Path(__file__).read_bytes()),
                  policy="original CIF/framework unchanged; BODYX guest FF only; source paths immutable",
                  gases={gas: {key: value for key, value in recipe.items()
                               if not key.startswith("source_")}
                         for gas, recipe in GASES.items()})
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.limit), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
