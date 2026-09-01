"""Order-preserving RDKit UFF parameters for all-atom initialization.

The numeric conventions reproduce RDKit UFF: Angstrom, kcal/mol, radians,
and force constants as returned by ``rdForceFieldHelpers``.  No Interchange
or OpenFF object is constructed here.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import numpy as np

from phantomwalk.lib.all_atom import AllAtomParameters, _molecular_compounds


def _type_name(prefix: str, values: tuple[Any, ...], table: dict) -> str:
    """Deduplicate a type by its complete coefficient tuple."""

    for name, current in table.items():
        if current["_key"] == values:
            return name
    name = f"{prefix}{len(table)}"
    table[name] = {"_key": values}
    return name


def _particle_atomic_number(particle: Any) -> int:
    element = getattr(particle, "element", None)
    number = getattr(element, "atomic_number", None)
    if number is None:
        raise ValueError(f"mBuild particle {particle!r} has no element")
    return int(number)


def _unsupported_atoms(mol: Any) -> str:
    return ", ".join(
        f"{atom.GetIdx()}:{atom.GetSymbol()}({atom.GetHybridization()})"
        for atom in mol.GetAtoms()
    )


def _angle_order(atom: Any, first: int, third: int, mol: Any) -> tuple[int, float | None]:
    """Return RDKit Builder.cpp's angle order and any ring-adjusted theta0."""

    from rdkit.Chem.rdchem import HybridizationType as H

    hybrid = atom.GetHybridization()
    if hybrid == H.SP:
        return 1, None
    if hybrid == H.SP2:
        rings = mol.GetRingInfo()
        center = atom.GetIdx()
        for size, outside_theta, inside_theta in ((3, 150.0, 60.0), (4, 135.0, 90.0)):
            if rings.IsAtomInRingOfSize(center, size):
                inside_first = rings.IsAtomInRingOfSize(first, size)
                inside_third = rings.IsAtomInRingOfSize(third, size)
                if inside_first != inside_third:
                    return 0, math.radians(outside_theta)
                if inside_first and inside_third:
                    return 0, math.radians(inside_theta)
        return 3, None
    if hybrid == H.SP3D2:
        return 4, None
    return 0, None


def _torsion_form(mol: Any, group: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return ``(periodicity, cosTerm)`` from RDKit TorsionAngle.cpp."""

    from rdkit.Chem.rdchem import HybridizationType as H

    i, j, k, ell = group
    aj, ak = mol.GetAtomWithIdx(j), mol.GetAtomWithIdx(k)
    hj, hk = aj.GetHybridization(), ak.GetHybridization()
    order, cos_term = 6, 1
    group6 = {8, 16, 34, 52, 84}
    bond_order = mol.GetBondBetweenAtoms(j, k).GetBondTypeAsDouble()
    if hj == H.SP3 and hk == H.SP3:
        order, cos_term = 3, -1
        if bond_order == 1.0 and aj.GetAtomicNum() in group6 and ak.GetAtomicNum() in group6:
            order, cos_term = 2, -1
    elif hj == H.SP2 and hk == H.SP2:
        order, cos_term = 2, 1
    elif bond_order == 1.0:
        sp3, sp2 = (aj, ak) if hj == H.SP3 else (ak, aj)
        if sp3.GetAtomicNum() in group6 and sp2.GetAtomicNum() not in group6:
            order, cos_term = 2, -1
        elif mol.GetAtomWithIdx(i).GetHybridization() == H.SP2 or mol.GetAtomWithIdx(ell).GetHybridization() == H.SP2:
            order, cos_term = 3, -1
    return order, cos_term


def _inversion_coefficients(atomic_number: int, carbonyl: bool) -> tuple[float, float, float]:
    """Return RDKit UFF C0/C1/C2 (Utils.cpp)."""

    if atomic_number in (6, 7, 8):
        return 1.0, -1.0, 0.0
    w0 = math.radians({15: 84.4339, 33: 86.9735, 51: 87.7047, 83: 90.0}[atomic_number])
    c2, c1 = 1.0, -4.0 * math.cos(w0)
    return -(c1 * math.cos(w0) + math.cos(2.0 * w0)), c1, c2


def parameterize_uff(compound: Any) -> AllAtomParameters:
    """Extract deterministic, HOOMD-ready UFF tables from an mBuild compound."""

    children = _molecular_compounds(compound)
    if len(children) != 1 or children[0] is not compound:
        return _parameterize_uff_children(compound, children)

    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers as uff

    mol = compound.to_rdkit()
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            bond.SetBondType(Chem.BondType.SINGLE)
    Chem.SanitizeMol(mol)
    particles = list(compound.particles())
    if mol.GetNumAtoms() != len(particles):
        raise ValueError(f"mBuild/RDKit atom-count mismatch: {len(particles)} != {mol.GetNumAtoms()}")
    mismatches = [
        (i, _particle_atomic_number(p), mol.GetAtomWithIdx(i).GetAtomicNum())
        for i, p in enumerate(particles)
        if _particle_atomic_number(p) != mol.GetAtomWithIdx(i).GetAtomicNum()
    ]
    if mismatches:
        raise ValueError(f"mBuild/RDKit atom-order mismatch: {mismatches}")
    if not uff.UFFHasAllMoleculeParams(mol):
        raise ValueError("UFF parameter coverage is incomplete; atoms/types: " + _unsupported_atoms(mol))
    box = getattr(compound, "box", None)
    if box is None:
        raise ValueError("compound.box must define periodic box lengths")
    masses = np.asarray([mol.GetAtomWithIdx(i).GetMass() for i in range(mol.GetNumAtoms())])
    result = AllAtomParameters(
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=np.asarray(box.lengths, dtype=float) * 10.0,
        masses_amu=masses,
    )

    particle_table: dict[str, dict[str, Any]] = {}
    epsilons, sigmas = [], []
    for i in range(mol.GetNumAtoms()):
        values = uff.GetUFFVdWParams(mol, i, i)
        if values is None:
            raise ValueError(f"UFF did not assign vdW parameters to atom {i}")
        sigma, epsilon = map(float, values)
        key = (sigma, epsilon, float(masses[i]))
        name = _type_name("uff_vdw_", key, particle_table)
        particle_table[name].update(sigma_a=sigma, epsilon_kcal_mol=epsilon, mass_amu=float(masses[i]))
        result.particle_types.append(name)
        sigmas.append(sigma); epsilons.append(epsilon)
    result.particle_type_params = {n: {k: v for k, v in p.items() if k != "_key"} for n, p in particle_table.items()}
    result.epsilon_ref_kcal_mol = max(epsilons)
    result.sigma_ref_a = max(sigmas)

    bond_table: dict[str, dict[str, Any]] = {}
    for bond in sorted(mol.GetBonds(), key=lambda b: tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))):
        group = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
        values = uff.GetUFFBondStretchParams(mol, *group)
        if values is None: raise ValueError(f"UFF did not assign bond {group}")
        kb, r0 = map(float, values); key = (kb, r0)
        name = _type_name("uff_bond_", key, bond_table)
        bond_table[name].update(k=kb, r0=r0)
        result.bonds.append(group); result.bond_types.append(name)
        result.bond_lengths_a[name] = r0
    result.bond_params = {n: {k: v for k, v in p.items() if k != "_key"} for n, p in bond_table.items()}

    angle_table: dict[str, dict[str, Any]] = {}
    for center in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(center)
        neighbors = sorted(n.GetIdx() for n in atom.GetNeighbors())
        for first, third in itertools.combinations(neighbors, 2):
            group = (first, center, third)
            values = uff.GetUFFAngleBendParams(mol, *group)
            if values is None: raise ValueError(f"UFF did not assign angle {group}")
            ka, theta0_degrees = map(float, values)
            theta0 = math.radians(theta0_degrees)
            order, adjusted = _angle_order(atom, first, third, mol)
            if adjusted is not None: theta0 = adjusted
            key = (ka, theta0, order)
            name = _type_name("uff_angle_", key, angle_table)
            angle_table[name].update(k=ka, t0=theta0, order=order)
            result.angles.append(group); result.angle_types.append(name)
    result.angle_params = {n: {k: v for k, v in p.items() if k != "_key"} for n, p in angle_table.items()}

    torsions_by_bond: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    for bond in sorted(mol.GetBonds(), key=lambda b: tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))):
        j, k = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        groups = []
        for i in sorted(n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k):
            for ell in sorted(n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j and n.GetIdx() != i):
                if uff.GetUFFTorsionParams(mol, i, j, k, ell) is not None:
                    groups.append((i, j, k, ell))
        if groups: torsions_by_bond[(j, k)] = groups
    torsion_table: dict[str, dict[str, Any]] = {}
    for groups in torsions_by_bond.values():
        divisor = len(groups)
        for group in groups:
            barrier = float(uff.GetUFFTorsionParams(mol, *group)) / divisor
            order, cos_term = _torsion_form(mol, group)
            key = (barrier, order, cos_term)
            name = _type_name("uff_torsion_", key, torsion_table)
            torsion_table[name].update(k=barrier, n=order, d=-cos_term, phi0=0.0)
            result.dihedrals.append(group); result.dihedral_types.append(name)
    result.dihedral_params = {n: {k: v for k, v in p.items() if k != "_key"} for n, p in torsion_table.items()}

    inversion_table: dict[str, dict[str, Any]] = {}
    for center in range(mol.GetNumAtoms()):
        neighbors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors())
        if len(neighbors) != 3: continue
        permutations = ((neighbors[0], center, neighbors[1], neighbors[2]), (neighbors[0], center, neighbors[2], neighbors[1]), (neighbors[1], center, neighbors[2], neighbors[0]))
        for group in permutations:
            kval = uff.GetUFFInversionParams(mol, *group)
            if kval is None: continue
            atomic_number = mol.GetAtomWithIdx(center).GetAtomicNum()
            carbonyl = atomic_number == 6 and any(mol.GetAtomWithIdx(n).GetAtomicNum() == 8 and mol.GetAtomWithIdx(n).GetHybridization() == Chem.HybridizationType.SP2 for n in neighbors)
            c0, c1, c2 = _inversion_coefficients(atomic_number, carbonyl)
            key = (float(kval), c0, c1, c2)
            name = _type_name("uff_inversion_", key, inversion_table)
            inversion_table[name].update(k=float(kval), c0=c0, c1=c1, c2=c2)
            result.impropers.append(group); result.improper_types.append(name)
    result.improper_params = {n: {k: v for k, v in p.items() if k != "_key"} for n, p in inversion_table.items()}
    return result


def _parameterize_uff_children(compound: Any, children: list[Any]) -> AllAtomParameters:
    """Parameterize disconnected chains once per unique molecular graph."""

    box = getattr(compound, "box", None)
    if box is None:
        raise ValueError("compound.box must define periodic box lengths")
    merged = AllAtomParameters(
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=np.asarray(box.lengths, dtype=float) * 10.0,
    )
    cached: dict[tuple[Any, ...], AllAtomParameters] = {}
    offset = 0
    for child in children:
        particles = list(child.particles())
        local_index = {particle: index for index, particle in enumerate(particles)}
        topology_key = (
            tuple(_particle_atomic_number(particle) for particle in particles),
            tuple(
                sorted(
                    tuple(sorted((local_index[first], local_index[second])))
                    for first, second in child.bonds()
                )
            ),
        )
        current = cached.get(topology_key)
        if current is None:
            original_box = child.box
            child.box = box
            try:
                current = parameterize_uff(child)
            finally:
                child.box = original_box
            cached[topology_key] = current
        for field_name in ("bonds", "angles", "dihedrals", "impropers"):
            getattr(merged, field_name).extend(
                tuple(index + offset for index in group)
                for group in getattr(current, field_name)
            )
        merged.masses_amu = np.concatenate((merged.masses_amu, current.masses_amu))
        table_pairs = (
            ("particle_types", "particle_type_params"),
            ("bond_types", "bond_params"),
            ("angle_types", "angle_params"),
            ("dihedral_types", "dihedral_params"),
            ("improper_types", "improper_params"),
        )
        for types_field, params_field in table_pairs:
            target_params = getattr(merged, params_field)
            mapping = {}
            for old_name, values in getattr(current, params_field).items():
                matching = next(
                    (name for name, existing in target_params.items() if existing == values),
                    None,
                )
                if matching is None:
                    matching = old_name
                    suffix = 1
                    while matching in target_params:
                        matching = f"{old_name}_{suffix}"
                        suffix += 1
                    target_params[matching] = values
                mapping[old_name] = matching
            getattr(merged, types_field).extend(
                mapping[name] for name in getattr(current, types_field)
            )
            if params_field == "bond_params":
                for old_name, length in current.bond_lengths_a.items():
                    merged.bond_lengths_a[mapping[old_name]] = length
        merged.epsilon_ref_kcal_mol = max(
            merged.epsilon_ref_kcal_mol, current.epsilon_ref_kcal_mol
        )
        merged.sigma_ref_a = max(merged.sigma_ref_a, current.sigma_ref_a)
        offset += len(particles)
    return merged


def uff_energy_components(parameters: AllAtomParameters, positions_a: np.ndarray) -> dict[str, float]:
    """Evaluate the extracted bonded UFF functional forms exactly.

    This reference evaluator is intentionally independent of HOOMD and is used
    to catch coefficient, unit, ordering, and functional-form regressions.
    """

    xyz = np.asarray(positions_a, dtype=float)

    def angle(group: tuple[int, int, int]) -> float:
        a, b, c = xyz[list(group)]
        u, v = a - b, c - b
        return math.acos(float(np.clip(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))

    def cos_dihedral(group: tuple[int, int, int, int]) -> float:
        p1, p2, p3, p4 = xyz[list(group)]
        t1 = np.cross(p1 - p2, p3 - p2)
        t2 = np.cross(p2 - p3, p4 - p3)
        norms = np.linalg.norm(t1) * np.linalg.norm(t2)
        return 0.0 if norms < 1e-20 else float(np.clip(np.dot(t1, t2) / norms, -1, 1))

    bond_e = sum(
        0.5 * parameters.bond_params[name]["k"]
        * (np.linalg.norm(xyz[first] - xyz[second]) - parameters.bond_params[name]["r0"]) ** 2
        for (first, second), name in zip(parameters.bonds, parameters.bond_types)
    )
    angle_e = 0.0
    for group, name in zip(parameters.angles, parameters.angle_types):
        values = parameters.angle_params[name]
        theta = angle(group); order = int(values["order"]); theta0 = values["t0"]
        if order:
            term = (1.0 - math.cos(order * theta)) / (order * order)
        else:
            c2 = 1.0 / (4.0 * max(math.sin(theta0) ** 2, 1e-8))
            c1 = -4.0 * c2 * math.cos(theta0)
            c0 = c2 * (2.0 * math.cos(theta0) ** 2 + 1.0)
            term = c0 + c1 * math.cos(theta) + c2 * math.cos(2.0 * theta)
        angle_e += values["k"] * term
        if order and order < 5 and math.cos(theta) > 0.8660:
            angle_e += math.exp(-20.0 * (theta - theta0 + 0.25))
    torsion_e = 0.0
    for group, name in zip(parameters.dihedrals, parameters.dihedral_types):
        values = parameters.dihedral_params[name]
        phi = math.acos(cos_dihedral(group))
        # Stored d follows HOOMD: d == -RDKit cosTerm.
        torsion_e += 0.5 * values["k"] * (1.0 + values["d"] * math.cos(values["n"] * phi))
    inversion_e = 0.0
    for group, name in zip(parameters.impropers, parameters.improper_types):
        i, j, k, ell = group
        ji, jk, jl = xyz[i] - xyz[j], xyz[k] - xyz[j], xyz[ell] - xyz[j]
        normal = np.cross(ji, jk)
        denominator = np.linalg.norm(normal) * np.linalg.norm(jl)
        cos_y = 0.0 if denominator < 1e-20 else float(np.clip(np.dot(normal, jl) / denominator, -1, 1))
        sin_y = math.sqrt(max(0.0, 1.0 - cos_y * cos_y))
        values = parameters.improper_params[name]
        inversion_e += values["k"] * (values["c0"] + values["c1"] * sin_y + values["c2"] * (2.0 * sin_y * sin_y - 1.0))
    # RDKit includes 1-4 and more distant graph pairs (Builder.cpp), with a
    # default cutoff of ten times the mixed UFF minimum.
    adjacency = [set() for _ in xyz]
    for first, second in parameters.bonds:
        adjacency[first].add(second); adjacency[second].add(first)
    graph_distance = np.full((len(xyz), len(xyz)), len(xyz) + 1, dtype=int)
    for root in range(len(xyz)):
        graph_distance[root, root] = 0
        pending = [root]
        while pending:
            current = pending.pop(0)
            for neighbor in adjacency[current]:
                if graph_distance[root, neighbor] > graph_distance[root, current] + 1:
                    graph_distance[root, neighbor] = graph_distance[root, current] + 1
                    pending.append(neighbor)
    vdw_e = 0.0
    for first in range(len(xyz)):
        p_first = parameters.particle_type_params[parameters.particle_types[first]]
        for second in range(first + 1, len(xyz)):
            if graph_distance[first, second] < 3:
                continue
            p_second = parameters.particle_type_params[parameters.particle_types[second]]
            xij = math.sqrt(p_first["sigma_a"] * p_second["sigma_a"])
            depth = math.sqrt(p_first["epsilon_kcal_mol"] * p_second["epsilon_kcal_mol"])
            distance = float(np.linalg.norm(xyz[first] - xyz[second]))
            if distance < 10.0 * xij:
                ratio6 = (xij / distance) ** 6
                vdw_e += depth * (ratio6 * ratio6 - 2.0 * ratio6)
    return {"bond": float(bond_e), "angle": angle_e, "torsion": torsion_e, "inversion": inversion_e, "vdw": vdw_e}
