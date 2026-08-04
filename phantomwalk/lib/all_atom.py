"""OpenFF parameter preparation for all-atom FastFIRE initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AllAtomParameters:
    """Numeric, HOOMD-ready parameters in OpenFF units.

    Lengths are in Angstrom and energies are in kcal/mol before reduction.
    """

    positions_a: np.ndarray
    box_lengths_a: np.ndarray
    bonds: list[tuple[int, int]] = field(default_factory=list)
    bond_types: list[str] = field(default_factory=list)
    bond_lengths_a: dict[str, float] = field(default_factory=dict)
    angles: list[tuple[int, int, int]] = field(default_factory=list)
    angle_types: list[str] = field(default_factory=list)
    angle_params: dict[str, dict[str, float]] = field(default_factory=dict)
    dihedrals: list[tuple[int, int, int, int]] = field(default_factory=list)
    dihedral_types: list[str] = field(default_factory=list)
    dihedral_params: dict[str, dict[str, float]] = field(default_factory=dict)
    impropers: list[tuple[int, int, int, int]] = field(default_factory=list)
    improper_types: list[str] = field(default_factory=list)
    improper_params: dict[str, dict[str, float]] = field(default_factory=dict)
    epsilon_ref_kcal_mol: float = 0.0
    sigma_ref_a: float = 0.0


def _openff_indices(key: Any) -> tuple[int, ...]:
    """Return atom indices from an OpenFF topology key."""

    indices = getattr(key, "atom_indices", key)
    return tuple(int(index) for index in indices)


def _as_float(value: Any, target_unit: Any) -> float:
    """Convert an OpenFF quantity to a plain float."""

    return float(value.m_as(target_unit))


def compound_to_openff_molecule(compound: Any) -> Any:
    """Convert an mBuild compound to an OpenFF molecule with its coordinates."""

    from openff.toolkit import Molecule
    from openff.units import unit

    rdkit_molecule = compound.to_rdkit()
    molecule = Molecule.from_rdkit(
        rdkit_molecule,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )
    if molecule.conformers:
        molecule._conformers.clear()
    molecule.add_conformer(np.asarray(compound.xyz, dtype=float) * unit.nanometer)
    return molecule


def parameterize_all_atom(
    compound: Any,
    force_field_name: str = "openff-2.3.0.offxml",
) -> AllAtomParameters:
    """Label an mBuild compound with Sage and return numeric parameter tables."""

    from openff.toolkit import ForceField, Topology
    from openff.units import unit

    particles = list(compound.particles())
    particle_index = {particle: index for index, particle in enumerate(particles)}
    bonds = [tuple(particle_index[particle] for particle in bond) for bond in compound.bonds()]
    molecule = compound_to_openff_molecule(compound)
    labels = ForceField(force_field_name).label_molecules(Topology.from_molecules([molecule]))[0]

    box = getattr(compound, "box", None)
    if box is None:
        raise ValueError("compound.box must define periodic box lengths")
    result = AllAtomParameters(
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=np.asarray(box.lengths, dtype=float) * 10.0,
        bonds=bonds,
    )

    epsilons = []
    sigmas = []
    for parameter in labels["vdW"].values():
        epsilons.append(_as_float(parameter.epsilon, unit.kilocalorie_per_mole))
        sigmas.append(_as_float(parameter.sigma, unit.angstrom))
    result.epsilon_ref_kcal_mol = max(epsilons)
    result.sigma_ref_a = max(sigmas)

    bond_labels = {_openff_indices(key): value for key, value in labels["Bonds"].items()}
    for bond in bonds:
        parameter = bond_labels.get(tuple(bond)) or bond_labels.get(tuple(reversed(bond)))
        if parameter is None:
            raise ValueError(f"Sage did not label bond {bond}")
        name = str(parameter.id)
        result.bond_types.append(name)
        result.bond_lengths_a[name] = _as_float(parameter.length, unit.angstrom)

    for key, parameter in labels["Angles"].items():
        name = str(parameter.id)
        result.angles.append(_openff_indices(key))
        result.angle_types.append(name)
        result.angle_params[name] = {
            "k": _as_float(parameter.k, unit.kilocalorie_per_mole / unit.radian**2),
            "t0": _as_float(parameter.angle, unit.radian),
        }

    _collect_torsions(labels["ProperTorsions"], result, improper=False, unit=unit)
    _collect_torsions(labels["ImproperTorsions"], result, improper=True, unit=unit)
    return result


def _collect_torsions(labels: dict[Any, Any], result: AllAtomParameters, improper: bool, unit: Any) -> None:
    """Expand multi-term OpenFF torsions into HOOMD periodic terms."""

    groups = result.impropers if improper else result.dihedrals
    types = result.improper_types if improper else result.dihedral_types
    params = result.improper_params if improper else result.dihedral_params
    phase_name = "chi0" if improper else "phi0"
    for key, parameter in labels.items():
        group = _openff_indices(key)
        idivf = getattr(parameter, "idivf", None) or [1.0] * len(parameter.k)
        for term, k_value in enumerate(parameter.k):
            name = f"{parameter.id}_{term}"
            signed_k = _as_float(k_value, unit.kilocalorie_per_mole) / float(idivf[term])
            groups.append(group)
            types.append(name)
            params[name] = {
                "k": abs(signed_k),
                "d": 1 if signed_k >= 0 else -1,
                "n": int(parameter.periodicity[term]),
                phase_name: _as_float(parameter.phase[term], unit.radian),
            }
