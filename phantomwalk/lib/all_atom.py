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


@dataclass(frozen=True)
class OpenMMMinimizationResult:
    """Energy and timing summary for an Interchange handoff validation."""

    initial_energy_kj_mol: float
    minimized_energy_kj_mol: float
    elapsed_s: float
    finite: bool


def _openff_indices(key: Any) -> tuple[int, ...]:
    """Return atom indices from an OpenFF topology key."""

    indices = getattr(key, "atom_indices", key)
    return tuple(int(index) for index in indices)


def _as_float(value: Any, target_unit: Any) -> float:
    """Convert an OpenFF quantity to a plain float."""

    return float(value.m_as(target_unit))


def compound_to_openff_molecule(compound: Any) -> Any:
    """Convert an mBuild compound to an OpenFF molecule with its coordinates."""

    from rdkit import Chem

    from openff.toolkit import Molecule
    from openff.units import unit

    rdkit_molecule = compound.to_rdkit()
    for bond in rdkit_molecule.GetBonds():
        if bond.GetBondType() == Chem.BondType.UNSPECIFIED:
            bond.SetBondType(Chem.BondType.SINGLE)
    Chem.SanitizeMol(rdkit_molecule)
    molecule = Molecule.from_rdkit(
        rdkit_molecule,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )
    if molecule.conformers:
        molecule._conformers.clear()
    molecule.add_conformer(np.asarray(compound.xyz, dtype=float) * unit.nanometer)
    return molecule


def _molecular_compounds(compound: Any) -> list[Any]:
    """Return disconnected top-level chains, or the compound itself."""

    children = list(compound.children)
    child_by_particle = {
        particle: child for child in children for particle in child.particles()
    }
    has_cross_child_bond = any(
        child_by_particle.get(first) is not child_by_particle.get(second)
        for first, second in compound.bonds()
    )
    if (
        len(children) > 1
        and sum(child.n_particles for child in children) == compound.n_particles
        and not has_cross_child_bond
    ):
        return children
    return [compound]


def parameterize_all_atom(
    compound: Any,
    force_field_name: str = "openff-2.3.0.offxml",
) -> AllAtomParameters:
    """Label an mBuild compound with Sage and return numeric parameter tables."""

    children = _molecular_compounds(compound)
    if len(children) != 1 or children[0] is not compound:
        return _parameterize_children(compound, children, force_field_name)

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


def create_interchange(
    compound: Any,
    force_field_name: str = "openff-2.3.0.offxml",
) -> Any:
    """Create an in-memory Sage Interchange with AshGC charges."""

    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Topology
    from openff.units import unit

    molecules = [compound_to_openff_molecule(child) for child in _molecular_compounds(compound)]
    topology = Topology.from_molecules(molecules)
    box = getattr(compound, "box", None)
    if box is None:
        raise ValueError("compound.box must define periodic box lengths")
    return Interchange.from_smirnoff(
        ForceField(force_field_name),
        topology,
        box=np.diag(np.asarray(box.lengths, dtype=float)) * unit.nanometer,
        positions=np.asarray(
            [particle.pos for particle in compound.particles()],
            dtype=float,
        )
        * unit.nanometer,
    )


def update_interchange_positions(interchange: Any, compound: Any) -> None:
    """Set Interchange positions directly from an initialized mBuild compound."""

    from openff.units import unit

    interchange.positions = (
        np.asarray([particle.pos for particle in compound.particles()], dtype=float)
        * unit.nanometer
    )


def update_compound_positions(compound: Any, interchange: Any) -> None:
    """Set mBuild coordinates from the current Interchange positions."""

    from openff.units import unit

    compound.xyz = np.asarray(interchange.positions.m_as(unit.nanometer), dtype=float)


def create_openmm_handoff(
    compound: Any,
    force_field_name: str = "openff-2.3.0.offxml",
) -> tuple[Any, Any]:
    """Create an Interchange and OpenMM System with Sage 2.3 AshGC charges."""

    interchange = create_interchange(compound, force_field_name)
    return interchange, interchange.to_openmm_system()


def minimize_interchange(
    interchange: Any,
    max_iterations: int = 0,
    platform_name: str = "CPU",
) -> OpenMMMinimizationResult:
    """Minimize an Interchange in OpenMM and retain the minimized coordinates.

    By default OpenMM runs until convergence; a positive iteration limit is
    intended only for short diagnostic tests.
    """

    import time

    import numpy as np
    import openmm
    from openmm import unit as openmm_unit
    from openff.units.openmm import from_openmm

    started = time.perf_counter()
    system = interchange.to_openmm_system()
    integrator = openmm.VerletIntegrator(0.0001 * openmm_unit.picoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(interchange.positions.to_openmm())
    initial_state = context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy().value_in_unit(
        openmm_unit.kilojoule_per_mole
    )
    openmm.LocalEnergyMinimizer.minimize(
        context,
        tolerance=10.0,
        maxIterations=max_iterations,
    )
    minimized_state = context.getState(getEnergy=True, getPositions=True)
    minimized_energy = minimized_state.getPotentialEnergy().value_in_unit(
        openmm_unit.kilojoule_per_mole
    )
    interchange.positions = from_openmm(minimized_state.getPositions(asNumpy=True))
    finite = bool(np.isfinite(initial_energy) and np.isfinite(minimized_energy))
    return OpenMMMinimizationResult(
        initial_energy_kj_mol=float(initial_energy),
        minimized_energy_kj_mol=float(minimized_energy),
        elapsed_s=time.perf_counter() - started,
        finite=finite,
    )


def _parameterize_children(
    compound: Any,
    children: list[Any],
    force_field_name: str,
) -> AllAtomParameters:
    """Parameterize disconnected top-level chains and merge their tables."""

    box = getattr(compound, "box", None)
    if box is None:
        raise ValueError("compound.box must define periodic box lengths")
    merged = AllAtomParameters(
        positions_a=np.asarray(compound.xyz, dtype=float) * 10.0,
        box_lengths_a=np.asarray(box.lengths, dtype=float) * 10.0,
    )
    offset = 0
    cached: dict[tuple[Any, ...], AllAtomParameters] = {}
    for child in children:
        particles = list(child.particles())
        local_index = {particle: index for index, particle in enumerate(particles)}
        topology_key = (
            tuple(particle.element.atomic_number for particle in particles),
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
            current = parameterize_all_atom(child, force_field_name)
            child.box = original_box
            cached[topology_key] = current
        merged.bonds.extend(
            tuple(index + offset for index in group) for group in current.bonds
        )
        merged.angles.extend(
            tuple(index + offset for index in group) for group in current.angles
        )
        merged.dihedrals.extend(
            tuple(index + offset for index in group) for group in current.dihedrals
        )
        merged.impropers.extend(
            tuple(index + offset for index in group) for group in current.impropers
        )
        merged.bond_types.extend(current.bond_types)
        merged.angle_types.extend(current.angle_types)
        merged.dihedral_types.extend(current.dihedral_types)
        merged.improper_types.extend(current.improper_types)
        merged.bond_lengths_a.update(current.bond_lengths_a)
        merged.angle_params.update(current.angle_params)
        merged.dihedral_params.update(current.dihedral_params)
        merged.improper_params.update(current.improper_params)
        merged.epsilon_ref_kcal_mol = max(
            merged.epsilon_ref_kcal_mol, current.epsilon_ref_kcal_mol
        )
        merged.sigma_ref_a = max(merged.sigma_ref_a, current.sigma_ref_a)
        offset += child.n_particles
    return merged


def _collect_torsions(
    labels: dict[Any, Any],
    result: AllAtomParameters,
    improper: bool,
    unit: Any,
) -> None:
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
