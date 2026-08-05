"""Build and run the all-atom systems used in the FastFIRE paper benchmarks."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, run_all_atom_fastfire

AMU_NM3_TO_G_CM3 = 1.66053906660e-3
SYSTEM_DENSITIES = {
    "pe": (0.2, 0.8, 1.1),
    "p3ht": (0.2, 0.8, 1.1),
    "pes": (0.2, 0.8, 1.3),
}
MONOMER_RESNAMES = {
    "pe": ("PE",),
    "p3ht": ("P3H",),
    "pes": ("BPA", "BPS"),
}


def _tagged_monomers(system: str) -> tuple[list[str], str]:
    """Return tagged-SMILES monomers and sequence for a test chemistry."""

    if system == "pe":
        return ["C{<}C{>}"], "A"
    if system == "p3ht":
        return ["c1{<}cc(sc1{>}CCCCCC)"], "A"
    if system == "pes":
        bisphenol_a = "c1{<}ccc(C(C)(C)c2ccc({>}cc2))cc1"
        diphenyl_sulfone = "O=S(=O)(c1{<}ccc(cc1))c1ccc({>}cc1)"
        # Three BPA and two BPS units give the requested 40:60 BPS:BPA ratio.
        return [bisphenol_a, diphenyl_sulfone], "AABAB"
    raise ValueError(f"unknown system {system!r}; choose from {tuple(SYSTEM_DENSITIES)}")


def build_chain(
    system: str,
    degree: int = 50,
    seed: int = 11,
    box_length: float | None = None,
) -> Any:
    """Build one explicit-hydrogen chain with mBuild's develop path API."""

    import mbuild as mb
    from mbuild.path import hard_sphere_random_walk
    from mbuild.path.constraints import CuboidConstraint

    smiles, sequence = _tagged_monomers(system)
    if degree % len(sequence):
        raise ValueError(f"degree must be divisible by the {len(sequence)}-unit sequence")
    polymer = mb.Polymer()
    for monomer_smiles, residue_name in zip(smiles, MONOMER_RESNAMES[system]):
        monomer = mb.load(monomer_smiles, smiles=True)
        monomer.name = residue_name
        polymer.add_monomer(monomer, head_tag="<", tail_tag=">", separation=0.15)
    constraint = None
    if box_length is not None:
        constraint = CuboidConstraint(box_length, pbc=(True, True, True))
    path = hard_sphere_random_walk(
        termination=degree,
        bond_length=0.30,
        radius=0.05,
        seed=seed,
        volume_constraint=constraint,
    )
    polymer.build_from_path(path, sequence=sequence, energy_minimize=False)
    return polymer


def build_test_system(
    system: str,
    density_g_cm3: float,
    target_atoms: int,
    degree: int = 50,
    seed: int = 11,
) -> Any:
    """Clone chains into a cubic box sized to the requested mass density."""

    import mbuild as mb

    chain = build_chain(system, degree, seed)
    n_chains = max(1, int(np.ceil(target_atoms / chain.n_particles)))
    chain_mass = sum(float(particle.mass) for particle in chain.particles())
    box_length = (n_chains * chain_mass * AMU_NM3_TO_G_CM3 / density_g_cm3) ** (1 / 3)
    root = mb.Compound(name=system.upper())
    for chain_index in range(n_chains):
        root.add(build_chain(system, degree, seed + chain_index, box_length))
    root.box = mb.Box(lengths=[box_length] * 3)
    return root


def _completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        keys.add((row["system"], row["density_g_cm3"], row["target_atoms"], row["seed"]))
    return keys


def write_visualization_pdb(compound: Any, path: Path) -> None:
    """Write a PDB with unique segments and one residue per mBuild monomer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    particles = list(compound.particles())
    if len(particles) > 99_999:
        raise ValueError("legacy PDB atom serials support at most 99,999 atoms")
    particle_index = {particle: index + 1 for index, particle in enumerate(particles)}
    chains = [child for child in compound.children if child.name == "Polymer"]
    if compound.name == "Polymer":
        chains = [compound]
    elif not chains or sum(chain.n_particles for chain in chains) != len(particles):
        chains = [compound]

    atom_locations: dict[Any, tuple[str, int, str]] = {}
    segment_index = 0
    for chain in chains:
        monomers = list(chain.children)
        if not monomers or sum(monomer.n_particles for monomer in monomers) != chain.n_particles:
            monomers = [chain]
        for monomer_index, monomer in enumerate(monomers):
            if monomer_index and monomer_index % 9_999 == 0:
                segment_index += 1
            encoded_segment = _base36(segment_index)
            if len(encoded_segment) > 4:
                raise ValueError("legacy PDB segment IDs support at most 36^4 segments")
            segment_id = encoded_segment.rjust(4, "0")
            residue_id = monomer_index % 9_999 + 1
            residue_name = str(monomer.name)[:3].upper()
            for particle in monomer.particles():
                atom_locations[particle] = (segment_id, residue_id, residue_name)
        segment_index += 1

    lengths_a = np.asarray(compound.box.lengths, dtype=float) * 10.0
    lines = [
        f"CRYST1{lengths_a[0]:9.3f}{lengths_a[1]:9.3f}{lengths_a[2]:9.3f}"
        "  90.00  90.00  90.00 P 1           1"
    ]
    atom_counts: dict[tuple[str, int, str], dict[str, int]] = {}
    for serial, particle in enumerate(particles, start=1):
        segment_id, residue_id, residue_name = atom_locations[particle]
        element = particle.element.symbol
        location = (segment_id, residue_id, residue_name)
        counts = atom_counts.setdefault(location, {})
        counts[element] = counts.get(element, 0) + 1
        atom_name = f"{element}{counts[element]}"[:4]
        x, y, z = np.asarray(particle.xyz, dtype=float)[0] * 10.0
        lines.append(
            f"HETATM{serial:5d} {atom_name:<4s} {residue_name:>3s}  {residue_id:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}      "
            f"{segment_id:<4s}{element:>2s}  "
        )
    for first, second in compound.bonds():
        lines.append(f"CONECT{particle_index[first]:5d}{particle_index[second]:5d}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def minimum_nonbonded_distance_a(compound: Any) -> float:
    """Return the closest periodic distance excluding 1-2, 1-3, and 1-4 pairs."""

    from scipy.spatial import cKDTree

    particles = list(compound.particles())
    particle_index = {particle: index for index, particle in enumerate(particles)}
    adjacency = [set() for _ in particles]
    for first, second in compound.bonds():
        i, j = particle_index[first], particle_index[second]
        adjacency[i].add(j)
        adjacency[j].add(i)
    excluded_codes = []
    n_particles = len(particles)
    for atom in range(len(particles)):
        seen = {atom}
        frontier = {atom}
        for _ in range(3):
            frontier = {neighbor for current in frontier for neighbor in adjacency[current]} - seen
            seen.update(frontier)
        excluded_codes.extend(
            min(atom, neighbor) * n_particles + max(atom, neighbor)
            for neighbor in seen
            if neighbor != atom
        )
    excluded_codes = np.unique(np.asarray(excluded_codes, dtype=np.int64))

    box = np.asarray(compound.box.lengths, dtype=float)
    positions = np.asarray([particle.pos for particle in particles], dtype=float) % box
    tree = cKDTree(positions, boxsize=box)
    closest = np.inf
    neighbor_count = min(32, n_particles)
    chunk_size = 10_000
    for start in range(0, n_particles, chunk_size):
        stop = min(start + chunk_size, n_particles)
        distances, neighbors = tree.query(
            positions[start:stop],
            k=neighbor_count,
            workers=-1,
        )
        atoms = np.arange(start, stop, dtype=np.int64)[:, None]
        candidate_codes = (
            np.minimum(atoms, neighbors) * n_particles + np.maximum(atoms, neighbors)
        )
        valid = ~np.isin(candidate_codes, excluded_codes, assume_unique=False)
        valid[:, 0] = False
        if np.any(valid):
            closest = min(closest, float(np.min(distances[valid])))
        unresolved = ~np.any(valid, axis=1)
        if np.any(unresolved) and neighbor_count < n_particles:
            extra_distances, extra_neighbors = tree.query(
                positions[start:stop][unresolved],
                k=min(128, n_particles),
                workers=-1,
            )
            extra_atoms = atoms[unresolved]
            extra_codes = (
                np.minimum(extra_atoms, extra_neighbors) * n_particles
                + np.maximum(extra_atoms, extra_neighbors)
            )
            extra_valid = ~np.isin(extra_codes, excluded_codes, assume_unique=False)
            extra_valid[:, 0] = False
            if not np.all(np.any(extra_valid, axis=1)):
                raise RuntimeError("could not locate a nonexcluded neighbor for every atom")
            closest = min(closest, float(np.min(extra_distances[extra_valid])))
    return closest * 10.0


def _base36(value: int) -> str:
    """Return a compact base-36 identifier for a non-negative integer."""

    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value < 0:
        raise ValueError("base-36 identifiers require non-negative integers")
    output = ""
    while value:
        value, remainder = divmod(value, 36)
        output = digits[remainder] + output
    return output or "0"


def run_matrix(
    output: Path,
    device: str = "auto",
    structures_dir: Path | None = None,
) -> None:
    """Run the restartable three-chemistry benchmark matrix as JSON Lines."""

    completed = _completed_keys(output)
    structures_dir = structures_dir or output.with_suffix("").with_name(
        f"{output.stem}_structures"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    for system, densities in SYSTEM_DENSITIES.items():
        for density in densities:
            for target_atoms in (10_000, 20_000, 40_000, 80_000):
                for seed in (11, 22, 33, 44, 55):
                    key = (system, density, target_atoms, seed)
                    pdb_path = structures_dir / (
                        f"{system}_rho{density:g}_n{target_atoms}_seed{seed}.pdb"
                    )
                    needs_structure = seed == 11 and not pdb_path.exists()
                    if key in completed and not needs_structure:
                        continue
                    construction_start = time.perf_counter()
                    compound = build_test_system(system, density, target_atoms, seed=seed)
                    construction_s = time.perf_counter() - construction_start
                    result = run_all_atom_fastfire(
                        compound,
                        AllAtomFastFIRESettings(seed=seed, device=device),
                    )
                    row = dict(
                        system=system,
                        density_g_cm3=density,
                        target_atoms=target_atoms,
                        actual_atoms=compound.n_particles,
                        seed=seed,
                        construction_s=construction_s,
                        minimum_nonbonded_distance_a=minimum_nonbonded_distance_a(compound),
                        **asdict(result),
                    )
                    if seed == 11:
                        write_visualization_pdb(compound, pdb_path)
                    if key in completed:
                        print(json.dumps({"structure_backfill": str(pdb_path)}), flush=True)
                        continue
                    with output.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("aa_fastfire_results.jsonl"))
    parser.add_argument("--device", choices=("auto", "CPU", "GPU"), default="auto")
    parser.add_argument("--structures-dir", type=Path)
    args = parser.parse_args()
    run_matrix(args.output, args.device, args.structures_dir)


if __name__ == "__main__":
    main()
