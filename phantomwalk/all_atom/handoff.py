"""Sage 2.3.0 hand-off minimization of FlowerMD-initialized coordinates.

The PhantomWalk contract is coordinates that a standard force field can
minimize at the target density. This module builds a periodic Sage 2.3.0
Interchange for an initialized mBuild compound, minimizes it with OpenMM,
and reports the energy the minimizer removed per atom, overall and by
force, in units of the largest Sage Lennard-Jones well depth (the paper's
quality metric: lower means closer to minimizer-ready).

Charges are Sage 2.3.0's own (NAGL AshGC model, library charges for
monatomic ions). Requires openff-toolkit, openff-interchange, openff-nagl
and openmm.
"""

import time

import numpy as np

SAGE = "openff-2.3.0.offxml"
KJ_PER_KCAL = 4.184


def build_sage_interchange(compound, box_nm, force_field=SAGE):
    """Return a periodic Interchange for `compound` at its current coordinates.

    Molecules are read with FlowerMD's charge-aware RDKit conversion and
    without stereochemistry, so every chain with the same graph is one
    OpenFF molecule type and NAGL charges one representative.
    """
    from flowermd.internal.all_atom_parameters import (
        compound_to_rdkit,
        molecular_compounds,
    )
    from openff.interchange import Interchange
    from openff.toolkit import ForceField, Molecule, Topology
    from openff.units import unit

    molecules = [
        Molecule.from_rdkit(
            compound_to_rdkit(child),
            allow_undefined_stereo=True,
            hydrogens_are_explicit=True,
        )
        for child in molecular_compounds(compound)
    ]
    return Interchange.from_smirnoff(
        ForceField(force_field),
        Topology.from_molecules(molecules),
        box=np.diag(np.asarray(box_nm, dtype=float)) * unit.nanometer,
        positions=np.asarray(compound.xyz, dtype=float) * unit.nanometer,
    )


def minimize_by_component(interchange, platform="CUDA", tolerance=10.0):
    """Minimize `interchange` with OpenMM and report energies by force.

    Parameters
    ----------
    interchange : openff.interchange.Interchange
    platform : str, default "CUDA"
        OpenMM platform; mixed precision on CUDA.
    tolerance : float, default 10.0
        `openmm.LocalEnergyMinimizer` force tolerance in kJ/mol/nm, with no
        iteration cap.

    Returns
    -------
    result : dict
        Energies (kJ/mol) before and after minimization, per force and
        total, timings and the final RMS force.
    positions_a : numpy.ndarray
        Minimized positions in Angstrom.

    """
    import openmm
    from openmm import unit

    system = interchange.to_openmm_system(combine_nonbonded_forces=False)
    names = {}
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)
        names[i] = f"{i}_{type(force).__name__}"
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(0.0001 * unit.picoseconds),
        openmm.Platform.getPlatformByName(platform),
        {"Precision": "mixed"} if platform == "CUDA" else {},
    )
    context.setPositions(interchange.positions.to_openmm())

    def by_force():
        return {
            name: float(
                context.getState(getEnergy=True, groups={i})
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            for i, name in names.items()
        }

    initial = by_force()
    started = time.perf_counter()
    openmm.LocalEnergyMinimizer.minimize(
        context, tolerance=tolerance, maxIterations=0
    )
    minimizer_s = time.perf_counter() - started
    final = by_force()
    state = context.getState(getPositions=True, getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer
    )
    total_i, total_f = sum(initial.values()), sum(final.values())
    return (
        {
            "platform": platform,
            "tolerance_kj_mol_nm": tolerance,
            "minimizer_s": minimizer_s,
            "initial_energy_kj_mol": total_i,
            "minimized_energy_kj_mol": total_f,
            "initial_by_force_kj_mol": initial,
            "minimized_by_force_kj_mol": final,
            "finite": bool(
                np.isfinite(total_i)
                and np.isfinite(total_f)
                and np.isfinite(forces).all()
            ),
            "final_rms_force_kj_mol_nm": float(np.sqrt(np.mean(forces**2))),
        },
        state.getPositions(asNumpy=True).value_in_unit(unit.angstrom),
    )


def sage_handoff(compound, box_nm, platform="CUDA", force_field=SAGE):
    """Build, minimize and score an initialized compound with Sage.

    Returns ``(result, minimized_positions_a)``. ``result`` adds to
    `minimize_by_component` the Interchange build time, the largest Sage
    epsilon, the net charge, and ``energy_removed_sage_epsilon_atom``, the
    energy removed per atom divided by that epsilon.
    """
    from openff.units import unit

    started = time.perf_counter()
    interchange = build_sage_interchange(compound, box_nm, force_field)
    interchange_s = time.perf_counter() - started
    eps_max = max(
        float(p.parameters["epsilon"].m_as("kilocalorie / mole"))
        for p in interchange.collections["vdW"].potentials.values()
    )
    net_charge = float(
        sum(
            q.m_as(unit.elementary_charge)
            for q in interchange.collections["Electrostatics"].charges.values()
        )
    )
    result, positions_a = minimize_by_component(interchange, platform=platform)
    n_atoms = compound.n_particles
    removed = (
        result["initial_energy_kj_mol"] - result["minimized_energy_kj_mol"]
    )
    result.update(
        {
            "force_field": force_field,
            "interchange_s": interchange_s,
            "epsilon_sage_max_kcal_mol": eps_max,
            "net_charge_e": net_charge,
            "energy_removed_kj_mol_atom": removed / n_atoms,
            "energy_removed_sage_epsilon_atom": removed
            / n_atoms
            / (KJ_PER_KCAL * eps_max),
        }
    )
    return result, positions_a
