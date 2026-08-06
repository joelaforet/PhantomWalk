"""Benchmark adaptive DPD/FIRE initialization and OpenMM handoff."""

from __future__ import annotations

import argparse
import json
import resource
import time
from dataclasses import asdict
from pathlib import Path

from phantomwalk.benchmarks.aa_test_systems import (
    build_test_system,
    minimum_nonbonded_distance_a,
)
from phantomwalk.lib.all_atom import (
    create_interchange,
    minimize_interchange,
    run_interchange_dynamics,
)
from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, run_all_atom_fastfire

HIGH_DENSITY = {"pe": 1.1, "p3ht": 1.1, "pes": 1.3}


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(","))


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_case(
    system: str,
    density: float,
    target_atoms: int,
    seed: int,
    measure_distance: bool,
    dynamics_steps: int,
    timestep_fs: float,
) -> dict:
    """Initialize and minimize one dense all-atom melt."""

    started = time.perf_counter()
    stage = time.perf_counter()
    compound = build_test_system(system, density, target_atoms, seed=seed)
    construction_s = time.perf_counter() - stage
    stage = time.perf_counter()
    interchange = create_interchange(compound)
    interchange_s = time.perf_counter() - stage
    fastfire = run_all_atom_fastfire(
        compound,
        AllAtomFastFIRESettings(device="CPU", seed=seed),
        interchange=interchange,
    )
    distance = minimum_nonbonded_distance_a(compound) if measure_distance else None
    minimization = minimize_interchange(interchange)
    dynamics = (
        run_interchange_dynamics(
            interchange,
            nvt_steps=dynamics_steps,
            npt_steps=dynamics_steps,
            report_interval=max(1, dynamics_steps // 10),
            timestep_fs=timestep_fs,
            seed=seed,
        )
        if dynamics_steps
        else None
    )
    return {
        "system": system,
        "density_g_cm3": density,
        "target_atoms": target_atoms,
        "actual_atoms": compound.n_particles,
        "seed": seed,
        "construction_s": construction_s,
        "interchange_s": interchange_s,
        "minimum_nonbonded_distance_a": distance,
        "openmm_initial_energy_kj_mol": minimization.initial_energy_kj_mol,
        "openmm_final_energy_kj_mol": minimization.minimized_energy_kj_mol,
        "openmm_initial_energy_per_atom": (
            minimization.initial_energy_kj_mol / compound.n_particles
        ),
        "openmm_final_energy_per_atom": (
            minimization.minimized_energy_kj_mol / compound.n_particles
        ),
        "openmm_s": minimization.elapsed_s,
        "openmm_platform": minimization.platform_name,
        "dynamics": asdict(dynamics) if dynamics is not None else None,
        "peak_rss_gib": _rss_gib(),
        "total_s": time.perf_counter() - started,
        **asdict(fastfire),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("aa_adaptive_scaling.jsonl"))
    parser.add_argument("--systems", type=_csv, default=("pe", "p3ht", "pes"))
    parser.add_argument("--target-atoms", type=int, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--density", type=float)
    parser.add_argument("--skip-distance", action="store_true")
    parser.add_argument("--dynamics-steps", type=int, default=0)
    parser.add_argument("--timestep-fs", type=float, default=2.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        density = args.density or HIGH_DENSITY[system]
        row = run_case(
            system,
            density,
            args.target_atoms,
            args.seed,
            not args.skip_distance,
            args.dynamics_steps,
            args.timestep_fs,
        )
        with args.output.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
