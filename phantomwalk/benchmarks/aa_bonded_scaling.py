"""Compare simple bonded-parameter scaling policies for dense AA melts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phantomwalk.benchmarks.aa_test_systems import (
    SYSTEM_DENSITIES,
    build_test_system,
    minimum_nonbonded_distance_a,
    write_visualization_pdb,
)
from phantomwalk.lib.all_atom import (
    create_interchange,
    minimize_interchange,
    parameterize_all_atom,
    update_compound_positions,
)
from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, run_all_atom_fastfire


def _bond_geometry(compound, parameters) -> dict[str, float]:
    """Summarize deviations from OpenFF equilibrium bond lengths."""

    positions_a = np.asarray(compound.xyz, dtype=float) * 10.0
    errors = []
    for (first, second), type_name in zip(
        parameters.bonds, parameters.bond_types
    ):
        length = np.linalg.norm(positions_a[second] - positions_a[first])
        errors.append(length - parameters.bond_lengths_a[type_name])
    return {
        "bond_rms_error_a": float(np.sqrt(np.mean(np.square(errors)))),
        "bond_max_abs_error_a": float(np.max(np.abs(errors))),
    }


def run_experiment(output: Path, structures_dir: Path, target_atoms: int) -> None:
    """Run both policies for every highest-density benchmark chemistry."""

    output.parent.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)
    for system, densities in SYSTEM_DENSITIES.items():
        density = max(densities)
        for mode in ("uniform", "class_normalized"):
            started = time.perf_counter()
            try:
                row = _run_case(
                    system, density, target_atoms, mode, structures_dir, started
                )
            except Exception as error:
                row = {
                    "system": system,
                    "density_g_cm3": density,
                    "target_atoms": target_atoms,
                    "mode": mode,
                    "stable": False,
                    "error_type": type(error).__name__,
                    "error": str(error).splitlines()[0],
                    "total_s": time.perf_counter() - started,
                }
            with output.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


def _run_case(system, density, target_atoms, mode, structures_dir, started):
    """Run and summarize one scaling-policy state point."""

    compound = build_test_system(system, density, target_atoms, seed=11)
    parameters = parameterize_all_atom(compound)
    interchange = create_interchange(compound)
    result = run_all_atom_fastfire(
        compound,
        AllAtomFastFIRESettings(
            bonded_parameterization=mode,
            device="CPU",
            seed=11,
        ),
        interchange=interchange,
    )
    fastfire_geometry = _bond_geometry(compound, parameters)
    fastfire_distance = minimum_nonbonded_distance_a(compound)
    stem = f"{system}_rho{density:g}_n{target_atoms}_{mode}"
    write_visualization_pdb(compound, structures_dir / f"{stem}_fastfire.pdb")
    minimization = minimize_interchange(interchange)
    update_compound_positions(compound, interchange)
    minimized_geometry = _bond_geometry(compound, parameters)
    write_visualization_pdb(compound, structures_dir / f"{stem}_minimized.pdb")
    return {
        "system": system,
        "density_g_cm3": density,
        "target_atoms": target_atoms,
        "actual_atoms": compound.n_particles,
        "mode": mode,
        "stable": True,
        "fastfire_s": result.elapsed_s,
        "fastfire_minimum_nonbonded_distance_a": fastfire_distance,
        **{f"fastfire_{key}": value for key, value in fastfire_geometry.items()},
        "openmm_initial_energy_kj_mol": minimization.initial_energy_kj_mol,
        "openmm_minimized_energy_kj_mol": minimization.minimized_energy_kj_mol,
        "openmm_minimization_s": minimization.elapsed_s,
        "openmm_finite": minimization.finite,
        "minimized_minimum_nonbonded_distance_a": minimum_nonbonded_distance_a(
            compound
        ),
        **{f"minimized_{key}": value for key, value in minimized_geometry.items()},
        "total_s": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("aa_bonded_scaling_results.jsonl")
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=Path("validation_structures/bonded_scaling"),
    )
    parser.add_argument("--target-atoms", type=int, default=10_000)
    args = parser.parse_args()
    run_experiment(args.output, args.structures_dir, args.target_atoms)


if __name__ == "__main__":
    main()
