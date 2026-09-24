"""Residue-aware structure and trajectory export for all-atom melts.

FlowerMD's placement systems hold a melt as an mBuild hierarchy, melt ->
molecule -> monomer, and every atom belongs to exactly one monomer (the
chain-end hydrogens sit in the terminal monomers). This module turns that
hierarchy into PDB records that MDAnalysis, PyMOL and NGLView all read the
same way:

* one residue per monomer, numbered from 1 within each molecule;
* one segment ID per molecule, four base-36 characters, so a box holds up to
  36**4 = 1,679,616 molecules. The chain ID column is left blank: it is a
  single character, and NGL takes the segment ID as the chain name when it
  is blank, so "chain" selections and colors still resolve per molecule;
* CONECT records for every bond and a CRYST1 record for the box;
* molecules made whole across the periodic boundary, each centroid in the
  0..L cell.

HOOMD and PDB both work in Angstrom, which is also FlowerMD's all-atom unit;
mBuild coordinates are in nm. Every function here takes Angstrom.
"""

import re
import warnings
from dataclasses import dataclass

import numpy as np

# Residue names by monomer name. Tactic copolymers (polystyrene, PMMA) name
# each residue by the enantiomer it was built from, read from the sequence
# FlowerMD appends to the chain name ("ps_20mer_AABAB"; A is the R monomer,
# B the S monomer). Polyethylene distinguishes the two chain ends.
RESIDUE_NAMES = {
    "polyethylene": "PEM",
    "ps": "PS",
    "pmma": "PMM",
    "pet": "ETP",
    "pc": "BPC",
    "pei": "PEI",
    "pim1": "PIM",
    "E": "ETH",
    "A": "ACR",
    "Na": "NA",
}
TERMINAL_RESIDUE_NAMES = {"polyethylene": "PET"}
TACTIC_RESIDUE_NAMES = {
    "ps": {"A": "PSR", "B": "PSS"},
    "pmma": {"A": "PMR", "B": "PMS"},
}

_BASE36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MAX_SEGMENTS = 36**4
MAX_ATOMS = 99_999
MAX_RESIDUES_PER_MOLECULE = 9_999


def segment_id(index):
    """Four-character base-36 segment ID of molecule `index` (0 -> "0000")."""
    if not 0 <= index < MAX_SEGMENTS:
        raise ValueError(
            f"PDB segment IDs hold at most {MAX_SEGMENTS} molecules."
        )
    digits = []
    for _ in range(4):
        index, digit = divmod(index, 36)
        digits.append(_BASE36[digit])
    return "".join(reversed(digits))


@dataclass
class Topology:
    """Per-atom PDB labels and bonds of a melt, in the compound's atom order.

    Attributes
    ----------
    names, elements, resnames, segids : (N,) str arrays
    resids : (N,) int array, residue number within the molecule, from 1
    molecule : (N,) int array, index of the molecule each atom belongs to
    bonds : (n_bonds, 2) int array of zero-based atom indices
    box_a : (3,) float array, orthorhombic box lengths in Angstrom
    """

    names: np.ndarray
    elements: np.ndarray
    resnames: np.ndarray
    resids: np.ndarray
    segids: np.ndarray
    molecule: np.ndarray
    bonds: np.ndarray
    box_a: np.ndarray

    @property
    def n_atoms(self):
        return len(self.names)

    @property
    def n_molecules(self):
        return int(self.molecule.max()) + 1 if self.n_atoms else 0


def _partition(compound, parts):
    """True when `parts` split `compound`'s particles exactly."""
    return bool(parts) and (
        sum(part.n_particles for part in parts) == compound.n_particles
    )


def _molecules(compound):
    children = list(compound.children)
    if compound.n_particles > 1 and _partition(compound, children):
        return children
    return [compound]


def _residues(molecule):
    children = list(molecule.children)
    if _partition(molecule, children):
        return children
    return [molecule]


def _sequence(molecule, n_residues):
    """Monomer letters FlowerMD appends to a tactic chain's name, or None."""
    match = re.search(r"_([AB]+)$", molecule.name)
    if match and len(match.group(1)) == n_residues:
        return match.group(1)
    return None


def _residue_name(monomer, index, n_residues, sequence, names):
    name = monomer.name
    if sequence is not None and name in TACTIC_RESIDUE_NAMES:
        return TACTIC_RESIDUE_NAMES[name][sequence[index]]
    if (
        name in TERMINAL_RESIDUE_NAMES
        and n_residues > 1
        and index in (0, n_residues - 1)
    ):
        return TERMINAL_RESIDUE_NAMES[name]
    if name in names:
        return names[name]
    return name[:3].upper()


def residue_topology(compound, box_a=None, residue_names=None):
    """Return the `Topology` of an mBuild melt, one residue per monomer.

    Parameters
    ----------
    compound : mbuild.Compound
        The melt, e.g. ``system.system`` of a FlowerMD placement, or the
        compound `AllAtomPhantomWalk.to_compound` returns. Atom order is
        ``compound.particles()``, the order FlowerMD simulates.
    box_a : (3,) array, optional
        Box lengths in Angstrom; default ``compound.box.lengths``.
    residue_names : dict, optional
        Monomer name -> residue name, merged over `RESIDUE_NAMES`. Tactic
        and chain-end names (`TACTIC_RESIDUE_NAMES`,
        `TERMINAL_RESIDUE_NAMES`) take precedence. Names are cut to three
        characters.
    """
    names = {**RESIDUE_NAMES, **(residue_names or {})}
    particles = list(compound.particles())
    index = {id(p): i for i, p in enumerate(particles)}
    n = len(particles)
    atom_names = np.empty(n, dtype=object)
    elements = np.empty(n, dtype=object)
    resnames = np.empty(n, dtype=object)
    resids = np.zeros(n, dtype=int)
    segids = np.empty(n, dtype=object)
    molecule = np.full(n, -1, dtype=int)

    for m, mol in enumerate(_molecules(compound)):
        segid = segment_id(m)
        residues = _residues(mol)
        if len(residues) > MAX_RESIDUES_PER_MOLECULE:
            raise ValueError(
                f"A molecule has {len(residues)} residues; PDB residue "
                f"numbers hold at most {MAX_RESIDUES_PER_MOLECULE}."
            )
        sequence = _sequence(mol, len(residues))
        for r, residue in enumerate(residues):
            resname = _residue_name(
                residue, r, len(residues), sequence, names
            )[:3]
            counts = {}
            for p in residue.particles():
                i = index[id(p)]
                symbol = p.element.symbol if p.element else p.name
                counts[symbol] = counts.get(symbol, 0) + 1
                atom_names[i] = f"{symbol}{counts[symbol]}"[:4]
                elements[i] = symbol
                resnames[i] = resname
                resids[i] = r + 1
                segids[i] = segid
                molecule[i] = m
    if (molecule < 0).any():
        raise ValueError("Some particles belong to no molecule.")

    bonds = np.asarray(
        [(index[id(a)], index[id(b)]) for a, b in compound.bonds()],
        dtype=int,
    ).reshape(-1, 2)
    if box_a is None:
        box_a = np.asarray(compound.box.lengths, dtype=float) * 10.0
    return Topology(
        names=atom_names,
        elements=elements,
        resnames=resnames,
        resids=resids,
        segids=segids,
        molecule=molecule,
        bonds=bonds,
        box_a=np.asarray(box_a, dtype=float),
    )


def _whole_offsets(positions, bonds, box):
    """Lattice offsets (in box lengths) that make every molecule whole.

    Each connected component is walked from its first atom with
    minimum-image bond vectors, as in the MuPT and FastFIRE exporters.
    """
    n = len(positions)
    adjacency = [[] for _ in range(n)]
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    whole = positions.copy()
    seen = np.zeros(n, dtype=bool)
    for root in range(n):
        if seen[root]:
            continue
        seen[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j in adjacency[i]:
                if seen[j]:
                    continue
                delta = positions[j] - positions[i]
                delta -= np.round(delta / box) * box
                whole[j] = whole[i] + delta
                seen[j] = True
                stack.append(j)
    return np.round((whole - positions) / box)


def _warn_if_wrapped(topology, positions):
    """Warn about molecules whose bonds cannot all be made short.

    A simulation measures each bond to the nearest image. If a bond was
    longer than half the box when it was first measured (a placement that
    stretches bonds between repeats), a ring of the molecule, e.g. a ladder
    junction, can close through a periodic image. No coordinates make such
    a molecule whole, and a bond in the written structure spans the box.
    """
    if not len(topology.bonds):
        return
    a, b = topology.bonds.T
    # after the walk, a bond longer than its minimum image closes a ring
    # through a periodic image
    delta = np.abs(positions[a] - positions[b])
    wrapped = (delta > 0.5 * topology.box_a + 1e-3).any(axis=1)
    if wrapped.any():
        molecules = sorted(set(topology.segids[a[wrapped]]))
        warnings.warn(
            f"{len(molecules)} molecule(s) ({', '.join(molecules[:10])}) "
            "close a ring through a periodic image and cannot be made "
            "whole; some of their bonds span the box. The structure itself "
            "is broken: the placement stretched a bond beyond half the box "
            "before the simulation measured it.",
            stacklevel=3,
        )


def _centroid_shift(positions, molecule, box):
    """Per-atom shift that puts each molecule's centroid in the 0..L cell."""
    n_mol = int(molecule.max()) + 1
    sums = np.zeros((n_mol, 3))
    np.add.at(sums, molecule, positions)
    centroids = sums / np.bincount(molecule, minlength=n_mol)[:, None]
    return (-np.floor(centroids / box) * box)[molecule]


def make_whole(topology, positions_a):
    """Positions with whole molecules, each centroid inside the 0..L cell."""
    box = topology.box_a
    positions = np.asarray(positions_a, dtype=float)
    positions = positions + _whole_offsets(positions, topology.bonds, box) * box
    _warn_if_wrapped(topology, positions)
    return positions + _centroid_shift(positions, topology.molecule, box)


def _atom_name_field(name, element):
    # PDB convention: a one-letter element starts in column 14.
    if len(name) < 4 and len(element) == 1:
        return f" {name:<3s}"
    return f"{name:<4s}"


def pdb_text(topology, positions_a, whole=True):
    """PDB text of one configuration (see `write_pdb`)."""
    if topology.n_atoms > MAX_ATOMS:
        raise ValueError(
            f"{topology.n_atoms} atoms: PDB atom serials and CONECT records "
            f"hold at most {MAX_ATOMS}."
        )
    positions = np.asarray(positions_a, dtype=float)
    if positions.shape != (topology.n_atoms, 3):
        raise ValueError(
            f"positions have shape {positions.shape}, expected "
            f"({topology.n_atoms}, 3)."
        )
    if whole:
        positions = make_whole(topology, positions)
    lx, ly, lz = topology.box_a
    lines = [
        f"CRYST1{lx:9.3f}{ly:9.3f}{lz:9.3f}  90.00  90.00  90.00 P 1           1"
    ]
    for i in range(topology.n_atoms):
        x, y, z = positions[i]
        element = topology.elements[i]
        lines.append(
            f"HETATM{i + 1:5d} "
            f"{_atom_name_field(topology.names[i], element)} "
            f"{topology.resnames[i]:>3s}  {topology.resids[i]:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}      "
            f"{topology.segids[i]:<4s}{element.upper():>2s}"
        )
    for a, b in topology.bonds:
        lines.append(f"CONECT{a + 1:5d}{b + 1:5d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(topology, positions_a, path, whole=True):
    """Write one configuration as a residue-aware PDB and return `path`.

    Parameters
    ----------
    topology : Topology
    positions_a : (N, 3) array
        Positions in Angstrom, wrapped or unwrapped, in topology order.
    path : str or path-like
    whole : bool, default True
        Make molecules whole and put each centroid in the box, which
        visualization and per-chain analysis such as Rg both need.
    """
    with open(path, "w") as handle:
        handle.write(pdb_text(topology, positions_a, whole=whole))
    return path


def frame_positions(frame):
    """Unwrapped positions (Angstrom) of a HOOMD frame, e.g. ``ff.frame``.

    ``AllAtomDPD.frame`` is the placement before DPD. Use this for it:
    `AllAtomPhantomWalk.to_compound` later overwrites the placement
    compound's coordinates in place.
    """
    position = np.asarray(frame.particles.position, dtype=float)
    image = frame.particles.image
    if image is None:
        return position
    box = np.asarray(frame.configuration.box[:3])
    return position + np.asarray(image) * box


def trajectory_positions(topology, gsd_path, stride=1, append_positions_a=None):
    """Continuous, whole-molecule frames of a HOOMD GSD trajectory.

    HOOMD stores wrapped positions plus image counts, so ``position + image
    * L`` follows every atom continuously. Frame 0 fixes which images make
    each molecule whole; every molecule then gets one fixed box shift,
    chosen so its centroid lies in the 0..L cell in the last frame. Chains
    therefore never jump between frames, which is what a movie needs.

    Parameters
    ----------
    topology : Topology
    gsd_path : str or path-like
        A trajectory written by FlowerMD's `Simulation` (``gsd_file_name``,
        every ``gsd_write_freq`` steps), in the topology's atom order.
    stride : int, default 1
        Keep every `stride`-th frame.
    append_positions_a : (N, 3) array, optional
        Unwrapped positions to add as a last frame, e.g.
        ``sim.final_positions() * 10`` after FIRE, when the writer's period
        did not land on the final step.

    Returns
    -------
    frames : (n_frames, N, 3) float32 array, Angstrom
    steps : (n_frames,) int array, simulation step of each frame (-1 for an
        appended frame)
    """
    import gsd.hoomd

    box = topology.box_a
    frames, steps = [], []
    with gsd.hoomd.open(str(gsd_path), "r") as trajectory:
        for frame in trajectory[::stride]:
            if frame.particles.N != topology.n_atoms:
                raise ValueError(
                    f"GSD frame has {frame.particles.N} particles, topology "
                    f"has {topology.n_atoms}."
                )
            if not np.allclose(frame.configuration.box[:3], box, rtol=1e-4):
                raise ValueError("GSD box differs from the topology box.")
            image = frame.particles.image
            if image is None:
                image = np.zeros((topology.n_atoms, 3))
            frames.append(frame.particles.position + image * box)
            steps.append(int(frame.configuration.step))
    if append_positions_a is not None:
        frames.append(np.asarray(append_positions_a, dtype=float))
        steps.append(-1)
    if not frames:
        raise ValueError(f"{gsd_path} holds no frames.")
    frames = np.asarray(frames, dtype=float)
    frames += _whole_offsets(frames[0], topology.bonds, box) * box
    _warn_if_wrapped(topology, frames[-1])
    frames += _centroid_shift(frames[-1], topology.molecule, box)
    return frames.astype(np.float32), np.asarray(steps, dtype=int)


def write_trajectory(
    topology, gsd_path, path, stride=1, append_positions_a=None
):
    """Convert a GSD trajectory to DCD or XTC for PyMOL, NGLView, MDAnalysis.

    The format follows the extension of `path`. Pair it with a PDB of the
    same topology (`write_pdb`) as the topology file. See
    `trajectory_positions` for the unwrapping and the arguments. Returns
    `path`.
    """
    import MDAnalysis as mda

    frames, _ = trajectory_positions(
        topology, gsd_path, stride=stride, append_positions_a=append_positions_a
    )
    universe = mda.Universe.empty(topology.n_atoms, trajectory=True)
    dimensions = np.r_[topology.box_a, 90.0, 90.0, 90.0]
    with mda.Writer(str(path), n_atoms=topology.n_atoms) as writer:
        for xyz in frames:
            universe.atoms.positions = xyz
            universe.dimensions = dimensions
            writer.write(universe.atoms)
    return path


def flush_trajectory(sim):
    """Flush the GSD writers of a HOOMD simulation to disk.

    HOOMD buffers GSD frames in memory; call this before reading the
    trajectory of a simulation that is still alive.
    """
    for writer in sim.operations.writers:
        flush = getattr(writer, "flush", None)
        if flush is not None:
            flush()


def universe(pdb_path, trajectory_path=None):
    """MDAnalysis Universe from an exported PDB and optional trajectory.

    Bonds come from the CONECT records, residues are monomers and segments
    are molecules, so ``u.segments`` iterates chains and
    ``u.select_atoms("resname PSR")`` picks R monomers.
    """
    import MDAnalysis as mda

    args = [str(pdb_path)] + ([str(trajectory_path)] if trajectory_path else [])
    return mda.Universe(*args)
