"""P1 CIF atom-order parser matching the original Vext input generator."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FORCEFIELD_PATH = Path(__file__).resolve().parents[1] / "inputs" / "forcefield" / "data_ff_UFF"


@dataclass(frozen=True)
class AtomSite:
    element: str
    fract_x: float
    fract_y: float
    fract_z: float
    charge: float


@dataclass(frozen=True)
class Cell:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float

    @property
    def matrix(self) -> np.ndarray:
        ar, br, gr = (math.radians(value) for value in (self.alpha, self.beta, self.gamma))
        volume_factor = math.sqrt(
            1 - math.cos(ar) ** 2 - math.cos(br) ** 2 - math.cos(gr) ** 2
            + 2 * math.cos(ar) * math.cos(br) * math.cos(gr)
        )
        return np.asarray([
            (self.a, 0.0, 0.0),
            (self.b * math.cos(gr), self.b * math.sin(gr), 0.0),
            (
                self.c * math.cos(br),
                self.c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / math.sin(gr),
                self.c * volume_factor / math.sin(gr),
            ),
        ], dtype=np.float64)


def _float_token(token: str) -> float:
    value = token.strip().strip("'\"").split("(")[0]
    if value in (".", "?"):
        raise ValueError(f"missing CIF numeric value: {token}")
    return float(value)


def parse_cif(path: Path) -> tuple[Cell, list[AtomSite]]:
    lines = Path(path).read_text().splitlines()
    cell_data: dict[str, float] = {}
    atoms: list[AtomSite] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        tokens = line.split()
        if tokens[0] in (
            "_cell_length_a", "_cell_length_b", "_cell_length_c",
            "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
        ) and len(tokens) >= 2:
            cell_data[tokens[0]] = _float_token(tokens[1])
            index += 1
            continue
        if tokens[0] != "loop_":
            index += 1
            continue

        index += 1
        tags = []
        while index < len(lines):
            tag = lines[index].strip()
            if not tag or tag.startswith("#"):
                index += 1
                continue
            if not tag.startswith("_"):
                break
            tags.append(tag.split()[0])
            index += 1
        positions = {tag: offset for offset, tag in enumerate(tags)}
        required = ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")
        if not all(tag in positions for tag in required):
            continue
        atom_type = positions.get("_atom_site_type_symbol", positions.get("_atom_site_label", 0))
        charge = positions.get("_atom_site_charge")
        needed = max(*(positions[tag] for tag in required), atom_type, charge or 0) + 1
        while index < len(lines):
            row_line = lines[index].strip()
            if not row_line or row_line.startswith("#"):
                index += 1
                continue
            if row_line.startswith(("loop_", "data_", "_")):
                break
            row = row_line.split()
            if len(row) >= needed:
                element_match = re.match(r"([A-Z][a-z]?)", row[atom_type])
                if element_match is None:
                    raise ValueError(f"bad element in {path}: {row[atom_type]}")
                atoms.append(AtomSite(
                    element=element_match.group(1),
                    fract_x=_float_token(row[positions[required[0]]]),
                    fract_y=_float_token(row[positions[required[1]]]),
                    fract_z=_float_token(row[positions[required[2]]]),
                    charge=_float_token(row[charge]) if charge is not None else 0.0,
                ))
            index += 1

    cell_keys = (
        "_cell_length_a", "_cell_length_b", "_cell_length_c",
        "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
    )
    if not atoms or any(key not in cell_data for key in cell_keys):
        raise ValueError(f"incomplete CIF: {path}")
    return Cell(*(cell_data[key] for key in cell_keys)), atoms


def load_uff_params(path: Path = FORCEFIELD_PATH) -> dict[str, tuple[float, float]]:
    values = {}
    with Path(path).open() as handle:
        for line in handle:
            tokens = line.split()
            if len(tokens) >= 4 and tokens[0] != "atom_type":
                values[tokens[0]] = float(tokens[1]), float(tokens[2])
    return values


def prepare_framework_inputs(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from ase.data import atomic_numbers

    cell, sites = parse_cif(path)
    params = load_uff_params()
    numbers = np.asarray([atomic_numbers[site.element] for site in sites], dtype=np.uint8)
    fractional = np.asarray(
        [(site.fract_x, site.fract_y, site.fract_z) for site in sites], dtype=np.float64
    )
    cartesian = fractional @ cell.matrix
    lengths = np.asarray([cell.a, cell.b, cell.c], dtype=np.float64)
    xyz = np.mod(cartesian / lengths, 1.0).astype(np.float32)
    forcefield = np.asarray([
        (*params[site.element], site.charge) for site in sites
    ], dtype=np.float32)
    cell_features = np.asarray([
        cell.a / 30, cell.b / 30, cell.c / 30,
        cell.alpha / 180, cell.beta / 180, cell.gamma / 180,
    ], dtype=np.float32)
    return numbers, xyz, forcefield, cell_features
