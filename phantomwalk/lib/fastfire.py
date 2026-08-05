"""Simple reduced-unit DPD and FIRE initialization for all-atom melts."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from phantomwalk.lib.all_atom import (
    AllAtomParameters,
    parameterize_all_atom,
    update_interchange_positions,
)


@dataclass(frozen=True)
class AllAtomFastFIRESettings:
    """Controls for the deliberately short all-atom FastFIRE protocol."""

    force_field: str = "openff-2.3.0.offxml"
    repulsion: float = 25_000.0
    bond_k: float = 250_000.0
    bonded_scale: float = 30.0
    gamma: float = 800.0
    r_cut: float = 1.01
    kT: float = 1.0
    dt: float = 0.0001
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
    seed: int = 1234
    device: str = "auto"
    nlist_exclusions: tuple[str, ...] = ("bond", "angle", "dihedral")


@dataclass(frozen=True)
class AllAtomFastFIREResult:
    """Timing and reduced-unit references from one FastFIRE run."""

    n_particles: int
    epsilon_ref_kcal_mol: float
    sigma_ref_a: float
    parameterization_s: float
    setup_s: float
    dpd_s: float
    dpd_steps: int
    dpd_converged: bool
    fire_s: float
    fire_steps: int
    fire_converged: bool
    bonded_counts: dict[str, int] = field(default_factory=dict)
    dpd_energy_history: list[dict[str, float]] = field(default_factory=list)
    dpd_energies: dict[str, float] = field(default_factory=dict)
    fire_energies: dict[str, float] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> float:
        """Return total measured wall time."""

        return self.parameterization_s + self.setup_s + self.dpd_s + self.fire_s


def _device(hoomd: Any, name: str) -> Any:
    if name.lower() == "cpu":
        return hoomd.device.CPU()
    if name.lower() == "gpu":
        return hoomd.device.GPU()
    if name.lower() != "auto":
        raise ValueError("device must be 'auto', 'CPU', or 'GPU'")
    return hoomd.device.auto_select()


def _set_groups(container: Any, groups: list[tuple[int, ...]], types: list[str]) -> None:
    unique_types = list(dict.fromkeys(types))
    container.N = len(groups)
    container.types = unique_types
    if groups:
        container.group = np.asarray(groups, dtype=np.uint32)
        container.typeid = np.asarray([unique_types.index(name) for name in types], dtype=np.uint32)


def _frame(parameters: AllAtomParameters) -> Any:
    import gsd.hoomd

    frame = gsd.hoomd.Frame()
    lengths = parameters.box_lengths_a / parameters.sigma_ref_a
    positions = parameters.positions_a / parameters.sigma_ref_a
    positions = (positions - lengths / 2 + lengths / 2) % lengths - lengths / 2
    frame.configuration.box = [*lengths, 0, 0, 0]
    frame.particles.N = len(positions)
    frame.particles.types = ["A"]
    frame.particles.typeid = np.zeros(len(positions), dtype=np.uint32)
    frame.particles.mass = np.ones(len(positions))
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
    forces = []
    if parameters.bonds:
        force = hoomd.md.bond.Harmonic()
        for name, r0 in parameters.bond_lengths_a.items():
            force.params[name] = {"k": settings.bond_k, "r0": r0 / parameters.sigma_ref_a}
        forces.append(force)
    if parameters.angles:
        force = hoomd.md.angle.Harmonic()
        for name, values in parameters.angle_params.items():
            force.params[name] = {
                "k": settings.bonded_scale
                * values["k"]
                / parameters.epsilon_ref_kcal_mol,
                "t0": values["t0"],
            }
        forces.append(force)
    if parameters.dihedrals:
        force = hoomd.md.dihedral.Periodic()
        for name, values in parameters.dihedral_params.items():
            force.params[name] = {
                **values,
                "k": settings.bonded_scale
                * values["k"]
                / parameters.epsilon_ref_kcal_mol,
            }
        forces.append(force)
    if parameters.impropers:
        force = hoomd.md.improper.Periodic()
        for name, values in parameters.improper_params.items():
            force.params[name] = {
                **values,
                "k": settings.bonded_scale
                * values["k"]
                / parameters.epsilon_ref_kcal_mol,
            }
        forces.append(force)
    nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=settings.nlist_exclusions)
    if conservative_dpd:
        dpd = hoomd.md.pair.DPDConservative(
            nlist, default_r_cut=settings.r_cut
        )
        dpd.params[("A", "A")] = {"A": settings.repulsion}
    else:
        dpd = hoomd.md.pair.DPD(
            nlist, default_r_cut=settings.r_cut, kT=settings.kT
        )
        dpd.params[("A", "A")] = {
            "A": settings.repulsion,
            "gamma": settings.gamma,
        }
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
        kind: energy / max(1, n_particles if kind == "pair" else bonded_counts[kind])
        for kind, energy in energies.items()
    }


def _energies_converged(
    previous: dict[str, float], current: dict[str, float], tolerance: float
) -> bool:
    """Return whether every intensive force energy changed by at most tolerance."""

    return all(
        abs(current[kind] - previous[kind]) / max(1.0, abs(previous[kind])) <= tolerance
        for kind in current
    )


def _unwrap(positions: np.ndarray, bonds: list[tuple[int, int]], box: np.ndarray) -> np.ndarray:
    adjacency = [[] for _ in positions]
    for first, second in bonds:
        adjacency[first].append(second)
        adjacency[second].append(first)
    output = positions.copy()
    visited: set[int] = set()
    for root in range(len(positions)):
        if root in visited:
            continue
        visited.add(root)
        pending = [root]
        while pending:
            atom = pending.pop()
            for neighbor in adjacency[atom]:
                if neighbor in visited:
                    continue
                delta = positions[neighbor] - positions[atom]
                delta -= box * np.rint(delta / box)
                output[neighbor] = output[atom] + delta
                visited.add(neighbor)
                pending.append(neighbor)
    return output


def run_all_atom_fastfire(
    compound: Any,
    settings: AllAtomFastFIRESettings | None = None,
    interchange: Any | None = None,
) -> AllAtomFastFIREResult:
    """Run DPD followed by FIRE to convergence and update ``compound``."""

    import hoomd

    settings = settings or AllAtomFastFIRESettings()
    started = time.perf_counter()
    parameters = parameterize_all_atom(compound, settings.force_field)
    parameterized = time.perf_counter()
    frame = _frame(parameters)
    forces = _forces(hoomd, parameters, settings)
    fire_forces = _forces(hoomd, parameters, settings, conservative_dpd=True)
    method = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    simulation = hoomd.Simulation(device=_device(hoomd, settings.device), seed=settings.seed)
    simulation.create_state_from_snapshot(frame)
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=settings.dt,
        forces=forces,
        methods=[method],
    )
    setup = time.perf_counter()
    simulation.run(settings.dpd_steps)
    dpd_steps = settings.dpd_steps
    bonded_counts = {
        "bond": len(parameters.bonds),
        "angle": len(parameters.angles),
        "dihedral": len(parameters.dihedrals),
        "improper": len(parameters.impropers),
    }
    dpd_energies = _force_energies(forces)
    previous = _intensive_energies(
        dpd_energies, len(parameters.positions_a), bonded_counts
    )
    dpd_energy_history = [{"step": float(dpd_steps), **previous}]
    stable_checks = 0
    while dpd_steps < settings.dpd_max_steps:
        interval = min(settings.dpd_interval, settings.dpd_max_steps - dpd_steps)
        simulation.run(interval)
        dpd_steps += interval
        dpd_energies = _force_energies(forces)
        current = _intensive_energies(
            dpd_energies, len(parameters.positions_a), bonded_counts
        )
        dpd_energy_history.append({"step": float(dpd_steps), **current})
        if _energies_converged(previous, current, settings.dpd_energy_tol):
            stable_checks += 1
            if stable_checks >= settings.dpd_consecutive_checks:
                break
        else:
            stable_checks = 0
        previous = current
    dpd_converged = stable_checks >= settings.dpd_consecutive_checks
    if settings.require_dpd_convergence and not dpd_converged:
        raise RuntimeError(
            f"DPD energies did not converge within {settings.dpd_max_steps} steps"
        )
    dpd_done = time.perf_counter()
    fire = hoomd.md.minimize.FIRE(
        dt=settings.dt,
        force_tol=settings.fire_force_tol,
        angmom_tol=1000.0,
        energy_tol=settings.fire_energy_tol,
        forces=fire_forces,
        methods=[method],
    )
    simulation.operations.integrator = fire
    simulation.run(settings.fire_steps)
    fire_steps = settings.fire_steps
    while not fire.converged and fire_steps < settings.fire_max_steps:
        interval = min(settings.fire_interval, settings.fire_max_steps - fire_steps)
        simulation.run(interval)
        fire_steps += interval
    fire_done = time.perf_counter()
    fire_energies = _force_energies(fire_forces)
    if settings.require_fire_convergence and not fire.converged:
        raise RuntimeError(
            f"FIRE did not converge within {settings.fire_max_steps} steps"
        )

    snapshot = simulation.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        reduced = np.asarray(snapshot.particles.position, dtype=float)
        box_reduced = parameters.box_lengths_a / parameters.sigma_ref_a
        reduced = _unwrap(reduced, parameters.bonds, box_reduced) + box_reduced / 2
        compound.xyz = reduced * parameters.sigma_ref_a / 10.0
        if interchange is not None:
            update_interchange_positions(interchange, compound)
    return AllAtomFastFIREResult(
        n_particles=len(parameters.positions_a),
        epsilon_ref_kcal_mol=parameters.epsilon_ref_kcal_mol,
        sigma_ref_a=parameters.sigma_ref_a,
        parameterization_s=parameterized - started,
        setup_s=setup - parameterized,
        dpd_s=dpd_done - setup,
        dpd_steps=dpd_steps,
        dpd_converged=dpd_converged,
        fire_s=fire_done - dpd_done,
        fire_steps=fire_steps,
        fire_converged=bool(fire.converged),
        bonded_counts=bonded_counts,
        dpd_energy_history=dpd_energy_history,
        dpd_energies=dpd_energies,
        fire_energies=fire_energies,
    )
