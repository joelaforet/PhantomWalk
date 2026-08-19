"""Physical-unit DPD and FIRE initialization for all-atom melts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from phantomwalk.lib.all_atom import (
    AllAtomParameters,
    make_molecules_whole,
    parameterize_all_atom,
    update_interchange_positions,
)


@dataclass(frozen=True)
class AllAtomFastFIRESettings:
    """Controls for the all-atom FastFIRE protocol in physical HOOMD units."""

    force_field: str = "openff-2.3.0.offxml"
    repulsion: float = 5_000.0
    bonded_scale: float = 30.0
    gamma: float = 800.0
    r_cut: float = 3.5
    kT: float = 1.0
    dt: float = 0.002
    dpd_steps: int = 2_000
    dpd_interval: int = 250
    dpd_max_steps: int = 10_000
    dpd_energy_tol: float = 0.02
    dpd_consecutive_checks: int = 2
    require_dpd_convergence: bool = True
    fire_steps: int = 100
    fire_interval: int = 200
    fire_max_steps: int = 10_000
    fire_force_tol: float = 1_000.0
    fire_energy_tol: float = 1_000.0
    require_fire_convergence: bool = True
    seed: int = 11
    device: str = "auto"
    nlist_exclusions: tuple[str, ...] = ("bond", "angle", "dihedral")


@dataclass(frozen=True)
class AllAtomFastFIREResult:
    """Timing, convergence, and parameter summary from one FastFIRE run."""

    n_particles: int
    epsilon_ref_kcal_mol: float
    parameterization_s: float
    setup_s: float
    dpd_s: float
    dpd_steps: int
    dpd_converged: bool
    fire_s: float
    fire_steps: int
    fire_converged: bool
    device_description: str
    bonded_counts: dict[str, int] = field(default_factory=dict)
    dpd_energy_history: list[dict[str, float | bool]] = field(default_factory=list)
    fire_energy_history: list[dict[str, float | bool]] = field(default_factory=list)
    dpd_energies: dict[str, float] = field(default_factory=dict)
    fire_energies: dict[str, float] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> float:
        """Return total measured wall time."""

        return self.parameterization_s + self.setup_s + self.dpd_s + self.fire_s


def _device(hoomd: Any, name: str) -> Any:
    """Create the requested HOOMD device."""

    normalized = name.lower()
    if normalized == "cpu":
        return hoomd.device.CPU()
    if normalized == "gpu":
        try:
            return hoomd.device.GPU()
        except RuntimeError as error:
            raise RuntimeError(
                "GPU requested, but this HOOMD build has GPU support disabled. "
                "Select the 'PhantomWalk GPU' Jupyter kernel or install the "
                "conda-forge HOOMD gpu build."
            ) from error
    if normalized != "auto":
        raise ValueError("device must be 'auto', 'CPU', or 'GPU'")
    return hoomd.device.auto_select()


def _device_description(device: Any) -> str:
    """Return a human-readable HOOMD device description."""

    label = device.__class__.__name__
    detail = getattr(device, "device", None)
    return f"{label}: {detail}" if detail else label


def _set_groups(container: Any, groups: list[tuple[int, ...]], types: list[str]) -> None:
    """Fill a HOOMD topology group with stable type IDs."""

    unique_types = sorted(set(types))
    container.N = len(groups)
    container.types = unique_types
    if groups:
        container.group = np.asarray(groups, dtype=np.uint32)
        container.typeid = np.asarray([unique_types.index(name) for name in types], dtype=np.uint32)


def _frame(parameters: AllAtomParameters) -> Any:
    """Create a physical Angstrom HOOMD GSD frame from all-atom parameters."""

    import gsd.hoomd

    positions = (parameters.positions_a % parameters.box_lengths_a) - parameters.box_lengths_a / 2.0
    particle_types = sorted(set(parameters.atom_types))
    frame = gsd.hoomd.Frame()
    frame.configuration.box = [*parameters.box_lengths_a.tolist(), 0, 0, 0]
    frame.particles.N = len(positions)
    frame.particles.types = particle_types
    frame.particles.typeid = np.asarray(
        [particle_types.index(atom_type) for atom_type in parameters.atom_types], dtype=np.uint32
    )
    frame.particles.mass = parameters.masses_amu
    frame.particles.position = positions
    _set_groups(frame.bonds, parameters.bonds, parameters.bond_types)
    _set_groups(frame.angles, parameters.angles, parameters.angle_types)
    _set_groups(frame.dihedrals, parameters.dihedrals, parameters.dihedral_types)
    _set_groups(frame.impropers, parameters.impropers, parameters.improper_types)
    return frame


def _forces(
    hoomd: Any,
    parameters: AllAtomParameters,
    settings: AllAtomFastFIRESettings,
    conservative_dpd: bool = False,
) -> list[Any]:
    """Create HOOMD forces in physical Angstrom and kcal/mol units."""

    forces = []
    if parameters.bonds:
        force = hoomd.md.bond.Harmonic()
        for name, values in parameters.bond_params.items():
            force.params[name] = {"k": settings.bonded_scale * values["k"], "r0": values["r0"]}
        forces.append(force)
    if parameters.angles:
        force = hoomd.md.angle.Harmonic()
        for name, values in parameters.angle_params.items():
            force.params[name] = {"k": settings.bonded_scale * values["k"], "t0": values["t0"]}
        forces.append(force)
    if parameters.dihedrals:
        force = hoomd.md.dihedral.Periodic()
        for name, values in parameters.dihedral_params.items():
            force.params[name] = {**values, "k": settings.bonded_scale * values["k"]}
        forces.append(force)
    if parameters.impropers:
        force = hoomd.md.improper.Periodic()
        for name, values in parameters.improper_params.items():
            force.params[name] = {**values, "k": settings.bonded_scale * values["k"]}
        forces.append(force)
    nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=settings.nlist_exclusions)
    if conservative_dpd:
        dpd = hoomd.md.pair.DPDConservative(nlist, default_r_cut=settings.r_cut)
    else:
        dpd = hoomd.md.pair.DPD(nlist, default_r_cut=settings.r_cut, kT=settings.kT)
    particle_types = sorted(set(parameters.atom_types))
    for type_i in particle_types:
        for type_j in particle_types:
            scale = (
                parameters.type_epsilons_kcal_mol[type_i]
                * parameters.type_epsilons_kcal_mol[type_j]
            ) ** 0.5 / parameters.epsilon_ref_kcal_mol
            values = {"A": settings.repulsion * scale}
            if not conservative_dpd:
                values["gamma"] = settings.gamma * scale
            dpd.params[(type_i, type_j)] = values
            dpd.r_cut[(type_i, type_j)] = settings.r_cut
    forces.append(dpd)
    return forces


def _force_energies(forces: list[Any]) -> dict[str, float]:
    """Return energies grouped by their bonded or pair-force role."""

    energies = {}
    for force in forces:
        module = force.__class__.__module__
        name = force.__class__.__name__
        if module.endswith(".bond") or ".bond." in module:
            kind = "bond"
        elif module.endswith(".angle") or ".angle." in module:
            kind = "angle"
        elif module.endswith(".dihedral") or ".dihedral." in module:
            kind = "dihedral"
        elif module.endswith(".improper") or ".improper." in module:
            kind = "improper"
        elif "DPD" in name:
            kind = "pair"
        else:
            continue
        energies[kind] = float(force.energy)
    return energies


def _intensive_energies(
    energies: dict[str, float],
    n_particles: int,
    bonded_counts: dict[str, int],
) -> dict[str, float]:
    """Normalize force energies by particles or bonded interactions."""

    return {
        f"{kind}_energy_{'per_atom' if kind == 'pair' else 'per_interaction'}": energy
        / max(1, n_particles if kind == "pair" else bonded_counts[kind])
        for kind, energy in energies.items()
    }


def _epsilon_per_atom_energies(
    energies: dict[str, float], n_particles: int, epsilon_ref: float
) -> dict[str, float]:
    """Normalize force energies by atom count and the largest vdW epsilon."""

    denominator = max(1, n_particles) * epsilon_ref
    output = {
        f"{kind}_energy_epsilon_per_atom": energy / denominator
        for kind, energy in energies.items()
    }
    output["total_energy_epsilon_per_atom"] = sum(energies.values()) / denominator
    return output


def _energies_converged(
    previous: dict[str, float], current: dict[str, float], tolerance: float
) -> bool:
    """Return whether every intensive force energy changed by at most tolerance."""

    return all(
        abs(current[kind] - previous[kind]) / max(1.0, abs(previous[kind])) <= tolerance
        for kind in current
    )


def _unwrap(positions: np.ndarray, bonds: list[tuple[int, int]], box: np.ndarray) -> np.ndarray:
    """Unwrap centered or wrapped coordinates by traversing the bond graph."""

    return make_molecules_whole(positions, bonds, box, center=False)


def _finite_snapshot(simulation: Any) -> bool:
    """Return whether the current HOOMD snapshot has finite coordinates."""

    snapshot = simulation.state.get_snapshot()
    if snapshot.communicator.rank != 0:
        return True
    return bool(np.all(np.isfinite(snapshot.particles.position)))


def _write_gsd(simulation: Any, path: str | Path | None) -> None:
    """Write a single-frame GSD if a destination was requested."""

    if path is None:
        return
    import hoomd

    hoomd.write.GSD.write(simulation.state, filename=str(path), mode="wb")


def _run_dpd(
    simulation: Any,
    forces: list[Any],
    parameters: AllAtomParameters,
    settings: AllAtomFastFIRESettings,
    bonded_counts: dict[str, int],
) -> tuple[int, bool, list[dict[str, float | bool]], dict[str, float]]:
    """Run DPD with baseline and block convergence checks."""

    dpd_steps = 0
    energies: dict[str, float] = {}
    previous: dict[str, float] | None = None
    history: list[dict[str, float | bool]] = []
    stable_checks = 0
    while dpd_steps < settings.dpd_max_steps:
        interval = min(settings.dpd_interval, settings.dpd_max_steps - dpd_steps)
        simulation.run(interval)
        dpd_steps += interval
        energies = _force_energies(forces)
        if not _finite_snapshot(simulation) or not all(np.isfinite(value) for value in energies.values()):
            raise RuntimeError("Non-finite HOOMD DPD coordinates or energies")
        current = _intensive_energies(energies, len(parameters.positions_a), bonded_counts)
        epsilon_per_atom = _epsilon_per_atom_energies(
            energies, len(parameters.positions_a), parameters.epsilon_ref_kcal_mol
        )
        passed = False
        if dpd_steps >= settings.dpd_steps:
            if previous is not None:
                passed = _energies_converged(previous, current, settings.dpd_energy_tol)
                stable_checks = stable_checks + 1 if passed else 0
            previous = current
        history.append(
            {
                "step": float(dpd_steps),
                **current,
                **epsilon_per_atom,
                "passed": passed,
            }
        )
        if stable_checks >= settings.dpd_consecutive_checks:
            break
    return dpd_steps, stable_checks >= settings.dpd_consecutive_checks, history, energies


def _run_fire(
    hoomd: Any,
    simulation: Any,
    forces: list[Any],
    settings: AllAtomFastFIRESettings,
    parameters: AllAtomParameters,
    bonded_counts: dict[str, int],
) -> tuple[int, bool, list[dict[str, float | bool]], dict[str, float]]:
    """Run FIRE minimum and continuation blocks with finite checks."""

    fire = hoomd.md.minimize.FIRE(
        dt=settings.dt,
        force_tol=settings.fire_force_tol,
        angmom_tol=1_000.0,
        energy_tol=settings.fire_energy_tol,
        forces=forces,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
    )
    simulation.operations.integrator = fire
    simulation.run(settings.fire_steps)
    fire_steps = settings.fire_steps
    energies = _force_energies(forces)
    if not _finite_snapshot(simulation) or not all(np.isfinite(value) for value in energies.values()):
        raise RuntimeError("Non-finite HOOMD FIRE coordinates or energies")
    intensive = _intensive_energies(energies, len(parameters.positions_a), bonded_counts)
    epsilon_per_atom = _epsilon_per_atom_energies(
        energies, len(parameters.positions_a), parameters.epsilon_ref_kcal_mol
    )
    history: list[dict[str, float | bool]] = [
        {
            "step": float(fire_steps),
            **intensive,
            **epsilon_per_atom,
            "passed": bool(fire.converged),
        }
    ]
    while not fire.converged and fire_steps < settings.fire_max_steps:
        interval = min(settings.fire_interval, settings.fire_max_steps - fire_steps)
        simulation.run(interval)
        fire_steps += interval
        energies = _force_energies(forces)
        if not _finite_snapshot(simulation) or not all(np.isfinite(value) for value in energies.values()):
            raise RuntimeError("Non-finite HOOMD FIRE coordinates or energies")
        intensive = _intensive_energies(energies, len(parameters.positions_a), bonded_counts)
        epsilon_per_atom = _epsilon_per_atom_energies(
            energies, len(parameters.positions_a), parameters.epsilon_ref_kcal_mol
        )
        history.append(
            {
                "step": float(fire_steps),
                **intensive,
                **epsilon_per_atom,
                "passed": bool(fire.converged),
            }
        )
    return fire_steps, bool(fire.converged), history, energies


def run_all_atom_fastfire(
    compound: Any,
    settings: AllAtomFastFIRESettings | None = None,
    interchange: Any | None = None,
    *,
    initial_gsd: str | Path | None = None,
    post_dpd_gsd: str | Path | None = None,
    post_fire_gsd: str | Path | None = None,
) -> AllAtomFastFIREResult:
    """Run DPD followed by FIRE to convergence and update ``compound``."""

    import hoomd

    settings = settings or AllAtomFastFIRESettings()
    started = time.perf_counter()
    parameters = parameterize_all_atom(compound, settings.force_field)
    parameterized = time.perf_counter()
    frame = _frame(parameters)
    forces = _forces(hoomd, parameters, settings)
    device = _device(hoomd, settings.device)
    simulation = hoomd.Simulation(device=device, seed=settings.seed)
    simulation.create_state_from_snapshot(frame)
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=settings.dt,
        forces=forces,
        methods=[hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())],
    )
    setup = time.perf_counter()
    _write_gsd(simulation, initial_gsd)
    bonded_counts = {
        "bond": len(parameters.bonds),
        "angle": len(parameters.angles),
        "dihedral": len(parameters.dihedrals),
        "improper": len(parameters.impropers),
    }

    dpd_steps, dpd_converged, dpd_history, dpd_energies = _run_dpd(
        simulation, forces, parameters, settings, bonded_counts
    )
    dpd_done = time.perf_counter()
    _write_gsd(simulation, post_dpd_gsd)
    if settings.require_dpd_convergence and not dpd_converged:
        raise RuntimeError(f"DPD energies did not converge within {settings.dpd_max_steps} steps")

    fire_forces = _forces(hoomd, parameters, settings, conservative_dpd=True)
    fire_steps, fire_converged, fire_history, fire_energies = _run_fire(
        hoomd, simulation, fire_forces, settings, parameters, bonded_counts
    )
    fire_done = time.perf_counter()
    _write_gsd(simulation, post_fire_gsd)
    if settings.require_fire_convergence and not fire_converged:
        raise RuntimeError(f"FIRE did not converge within {settings.fire_max_steps} steps")

    snapshot = simulation.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        centered_a = np.asarray(snapshot.particles.position, dtype=float)
        if not np.all(np.isfinite(centered_a)):
            raise RuntimeError("Non-finite final HOOMD coordinates")
        positions_a = _unwrap(centered_a + parameters.box_lengths_a / 2.0, parameters.bonds, parameters.box_lengths_a)
        compound.xyz = positions_a / 10.0
        if interchange is not None:
            update_interchange_positions(interchange, compound)
    return AllAtomFastFIREResult(
        n_particles=len(parameters.positions_a),
        epsilon_ref_kcal_mol=parameters.epsilon_ref_kcal_mol,
        parameterization_s=parameterized - started,
        setup_s=setup - parameterized,
        dpd_s=dpd_done - setup,
        dpd_steps=dpd_steps,
        dpd_converged=dpd_converged,
        fire_s=fire_done - dpd_done,
        fire_steps=fire_steps,
        fire_converged=fire_converged,
        device_description=_device_description(device),
        bonded_counts=bonded_counts,
        dpd_energy_history=dpd_history,
        fire_energy_history=fire_history,
        dpd_energies=dpd_energies,
        fire_energies=fire_energies,
    )
