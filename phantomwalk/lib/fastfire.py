"""Simple reduced-unit DPD and FIRE initialization for all-atom melts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from phantomwalk.lib.all_atom import AllAtomParameters, parameterize_all_atom


@dataclass(frozen=True)
class AllAtomFastFIRESettings:
    """Controls for the deliberately short all-atom FastFIRE protocol."""

    force_field: str = "openff-2.3.0.offxml"
    repulsion: float = 250_000.0
    bond_k: float = 250_000.0
    gamma: float = 1_500.0
    r_cut: float = 1.01
    kT: float = 1.0
    dt: float = 0.0001
    dpd_steps: int = 500
    fire_steps: int = 200
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
    fire_s: float

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
                "k": values["k"] / parameters.epsilon_ref_kcal_mol,
                "t0": values["t0"],
            }
        forces.append(force)
    if parameters.dihedrals:
        force = hoomd.md.dihedral.Periodic()
        for name, values in parameters.dihedral_params.items():
            force.params[name] = {
                **values,
                "k": values["k"] / parameters.epsilon_ref_kcal_mol,
            }
        forces.append(force)
    if parameters.impropers:
        force = hoomd.md.improper.Periodic()
        for name, values in parameters.improper_params.items():
            force.params[name] = {
                **values,
                "k": values["k"] / parameters.epsilon_ref_kcal_mol,
            }
        forces.append(force)
    nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=settings.nlist_exclusions)
    dpd = hoomd.md.pair.DPD(nlist, default_r_cut=settings.r_cut, kT=settings.kT)
    dpd.params[("A", "A")] = {"A": settings.repulsion, "gamma": settings.gamma}
    forces.append(dpd)
    return forces


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
) -> AllAtomFastFIREResult:
    """Run 500 DPD steps followed by 200 FIRE steps and update ``compound``."""

    import hoomd

    settings = settings or AllAtomFastFIRESettings()
    started = time.perf_counter()
    parameters = parameterize_all_atom(compound, settings.force_field)
    parameterized = time.perf_counter()
    frame = _frame(parameters)
    forces = _forces(hoomd, parameters, settings)
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
    dpd_done = time.perf_counter()
    simulation.operations.integrator = hoomd.md.minimize.FIRE(
        dt=settings.dt,
        force_tol=0.1,
        angmom_tol=1000.0,
        energy_tol=0.1,
        forces=forces,
        methods=[method],
    )
    simulation.run(settings.fire_steps)
    fire_done = time.perf_counter()

    snapshot = simulation.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        reduced = np.asarray(snapshot.particles.position, dtype=float)
        box_reduced = parameters.box_lengths_a / parameters.sigma_ref_a
        reduced = _unwrap(reduced, parameters.bonds, box_reduced) + box_reduced / 2
        compound.xyz = reduced * parameters.sigma_ref_a / 10.0
    return AllAtomFastFIREResult(
        n_particles=len(parameters.positions_a),
        epsilon_ref_kcal_mol=parameters.epsilon_ref_kcal_mol,
        sigma_ref_a=parameters.sigma_ref_a,
        parameterization_s=parameterized - started,
        setup_s=setup - parameterized,
        dpd_s=dpd_done - setup,
        fire_s=fire_done - dpd_done,
    )
