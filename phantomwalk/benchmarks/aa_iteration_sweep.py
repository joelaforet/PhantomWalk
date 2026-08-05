"""Sweep fixed DPD and FIRE step counts for dense all-atom melts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from phantomwalk.benchmarks.aa_test_systems import (
    build_test_system,
    minimum_nonbonded_distance_a,
)
from phantomwalk.lib.all_atom import (
    create_interchange,
    minimize_interchange,
    parameterize_all_atom,
)
from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, run_all_atom_fastfire

HIGH_DENSITY = {"pe": 1.1, "p3ht": 1.1, "pes": 1.3}


def _parse_steps(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def _minimum_endpoint_distance_a(compound, groups) -> float:
    positions = np.asarray(compound.xyz, dtype=float) * 10.0
    box = np.asarray(compound.box.lengths, dtype=float) * 10.0
    delta = positions[[group[-1] for group in groups]] - positions[
        [group[0] for group in groups]
    ]
    delta -= box * np.rint(delta / box)
    return float(np.min(np.linalg.norm(delta, axis=1)))


def _completed(path: Path) -> set[tuple[str, int, int, int, int]]:
    if not path.exists():
        return set()
    return {
        (
            row["system"],
            row["target_atoms"],
            row["seed"],
            row["dpd_steps"],
            row["fire_steps"],
        )
        for row in map(json.loads, path.read_text().splitlines())
    }


def run_sweep(
    output: Path,
    target_atoms: int,
    dpd_steps: tuple[int, ...],
    fire_steps: tuple[int, ...],
    seed: int,
) -> None:
    """Run the Cartesian iteration grid and append one JSON row per case."""

    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed(output)
    for system, density in HIGH_DENSITY.items():
        for n_dpd in dpd_steps:
            for n_fire in fire_steps:
                key = (system, target_atoms, seed, n_dpd, n_fire)
                if key in completed:
                    continue
                started = time.perf_counter()
                base = {
                    "system": system,
                    "density_g_cm3": density,
                    "target_atoms": target_atoms,
                    "degree": 50,
                    "seed": seed,
                    "dpd_steps": n_dpd,
                    "fire_steps": n_fire,
                }
                try:
                    compound = build_test_system(
                        system, density, target_atoms, degree=50, seed=seed
                    )
                    interchange = create_interchange(compound)
                    parameters = parameterize_all_atom(compound)
                    result = run_all_atom_fastfire(
                        compound,
                        AllAtomFastFIRESettings(
                            device="CPU",
                            seed=seed,
                            dpd_steps=n_dpd,
                            dpd_max_steps=n_dpd,
                            require_dpd_convergence=False,
                            fire_steps=n_fire,
                            fire_max_steps=n_fire,
                            require_fire_convergence=False,
                        ),
                        interchange=interchange,
                    )
                    nonbonded_distance = minimum_nonbonded_distance_a(compound)
                    one_four_distance = _minimum_endpoint_distance_a(
                        compound, parameters.dihedrals
                    )
                    minimization = minimize_interchange(interchange)
                    initial_per_atom = (
                        minimization.initial_energy_kj_mol / compound.n_particles
                    )
                    final_per_atom = (
                        minimization.minimized_energy_kj_mol / compound.n_particles
                    )
                    sensible = bool(
                        minimization.finite
                        and minimization.minimized_energy_kj_mol
                        < minimization.initial_energy_kj_mol
                        and nonbonded_distance >= 0.75
                        and one_four_distance >= 0.35
                        and initial_per_atom <= 1.0e7
                        and final_per_atom <= 10.0
                    )
                    row = {
                        **base,
                        "actual_atoms": compound.n_particles,
                        "stable": True,
                        "sensible": sensible,
                        "fastfire_s": result.elapsed_s,
                        "dpd_s": result.dpd_s,
                        "fire_s": result.fire_s,
                        "fire_converged": result.fire_converged,
                        "minimum_nonbonded_distance_a": nonbonded_distance,
                        "minimum_one_four_distance_a": one_four_distance,
                        "dpd_energies": result.dpd_energies,
                        "fire_energies": result.fire_energies,
                        "openmm_platform": minimization.platform_name,
                        "openmm_initial_energy_kj_mol": (
                            minimization.initial_energy_kj_mol
                        ),
                        "openmm_final_energy_kj_mol": (
                            minimization.minimized_energy_kj_mol
                        ),
                        "openmm_initial_energy_per_atom": initial_per_atom,
                        "openmm_final_energy_per_atom": final_per_atom,
                        "openmm_s": minimization.elapsed_s,
                        "protocol_s": result.elapsed_s + minimization.elapsed_s,
                        "total_s": time.perf_counter() - started,
                    }
                except Exception as error:
                    row = {
                        **base,
                        "stable": False,
                        "sensible": False,
                        "error_type": type(error).__name__,
                        "error": str(error).splitlines()[0],
                        "total_s": time.perf_counter() - started,
                    }
                with output.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("aa_iteration_sweep.jsonl")
    )
    parser.add_argument("--target-atoms", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--dpd-steps", type=_parse_steps, default=(250, 500, 1000, 2000))
    parser.add_argument("--fire-steps", type=_parse_steps, default=(50, 100, 200, 400))
    args = parser.parse_args()
    run_sweep(args.output, args.target_atoms, args.dpd_steps, args.fire_steps, args.seed)


if __name__ == "__main__":
    main()
