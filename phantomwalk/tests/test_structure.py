import warnings

import numpy as np
import pytest

pytest.importorskip("flowermd")
pytest.importorskip("rdkit")
mda = pytest.importorskip("MDAnalysis")

import unyt as u  # noqa: E402

from phantomwalk.all_atom.structure import (  # noqa: E402
    MAX_SEGMENTS,
    make_whole,
    residue_topology,
    segment_id,
    trajectory_positions,
    universe,
    write_pdb,
    write_trajectory,
)


def _melt(chains, density=0.8, unit="repeat"):
    from flowermd.library import AllAtomLattice

    return AllAtomLattice(
        chains, density=density * u.g / u.cm**3, unit=unit, seed=1
    ).system


def _max_bond(topology, positions):
    a, b = topology.bonds.T
    return np.linalg.norm(positions[a] - positions[b], axis=1).max()


def _max_bond_min_image(topology, positions):
    a, b = topology.bonds.T
    delta = positions[a] - positions[b]
    delta -= np.round(delta / topology.box_a) * topology.box_a
    return np.linalg.norm(delta, axis=1).max()


def test_segment_ids():
    assert segment_id(0) == "0000"
    assert segment_id(35) == "000Z"
    assert segment_id(36) == "0010"
    assert segment_id(MAX_SEGMENTS - 1) == "ZZZZ"
    ids = {segment_id(i) for i in range(2000)}
    assert len(ids) == 2000
    with pytest.raises(ValueError):
        segment_id(MAX_SEGMENTS)


def test_polyethylene_residues():
    from flowermd.library import PolyEthylene

    compound = _melt(PolyEthylene(lengths=5, num_mols=3))
    top = residue_topology(compound)
    assert top.n_atoms == compound.n_particles
    assert top.n_molecules == 3
    for m in range(3):
        mol = top.molecule == m
        assert set(top.segids[mol]) == {segment_id(m)}
        assert sorted(set(top.resids[mol])) == [1, 2, 3, 4, 5]
        ends = top.resnames[mol & np.isin(top.resids, [1, 5])]
        middle = top.resnames[mol & np.isin(top.resids, [2, 3, 4])]
        assert set(ends) == {"PET"} and set(middle) == {"PEM"}


def test_tactic_residue_names_follow_sequence():
    from flowermd.library import PolyStyrene

    compound = _melt(
        PolyStyrene(lengths=6, num_mols=4, tacticity="atactic", seed=3),
        unit="chain",
    )
    top = residue_topology(compound)
    for m, mol in enumerate(compound.children):
        sequence = mol.name.rsplit("_", 1)[1]
        names = [
            top.resnames[(top.molecule == m) & (top.resids == r + 1)][0]
            for r in range(6)
        ]
        assert names == [{"A": "PSR", "B": "PSS"}[s] for s in sequence]


def test_ionomer_and_counterions():
    from flowermd.library import PEAAIonomer

    chains = PEAAIonomer(
        lengths=1, num_mols=3, pattern="EEAEE", tacticity="atactic", seed=5
    )
    compound = _melt([chains, chains.counterions("[Na+]")])
    top = residue_topology(compound)
    assert set(top.resnames) == {"ETH", "ACR", "NA"}
    sodium = top.elements == "Na"
    assert sodium.sum() == 3
    # every ion is its own molecule and residue
    assert len(set(top.segids[sodium])) == 3


def test_pdb_round_trip(tmp_path):
    from flowermd.library import PolyStyrene

    compound = _melt(
        PolyStyrene(lengths=4, num_mols=5, tacticity="atactic", seed=2),
        unit="chain",
    )
    top = residue_topology(compound)
    path = write_pdb(top, compound.xyz * 10, tmp_path / "melt.pdb")
    melt = universe(path)
    assert melt.atoms.n_atoms == top.n_atoms
    assert melt.segments.n_segments == 5
    assert melt.residues.n_residues == 20
    assert len(melt.bonds) == len(top.bonds)
    assert len(melt.atoms.fragments) == 5
    assert np.allclose(melt.dimensions[:3], top.box_a, atol=1e-3)
    assert list(melt.atoms.elements[:2]) == list(top.elements[:2])


def test_make_whole_across_boundary():
    from flowermd.library import PolyEthylene

    compound = _melt(PolyEthylene(lengths=8, num_mols=2), density=0.2)
    top = residue_topology(compound)
    box = top.box_a
    longest = _max_bond_min_image(top, compound.xyz * 10)
    whole = make_whole(top, compound.xyz * 10)
    wrapped = np.mod(whole + 0.4 * box, box)  # split chains at the boundary
    assert _max_bond(top, wrapped) > 0.5 * box.min()
    rebuilt = make_whole(top, wrapped)
    assert _max_bond(top, rebuilt) == pytest.approx(longest, abs=1e-6)
    for m in range(top.n_molecules):
        centroid = rebuilt[top.molecule == m].mean(axis=0)
        assert np.all((centroid >= 0) & (centroid < box))


def test_gsd_trajectory_is_continuous(tmp_path):
    import gsd.hoomd
    from flowermd.library import PolyEthylene

    compound = _melt(PolyEthylene(lengths=8, num_mols=3), density=0.2)
    top = residue_topology(compound)
    box = top.box_a
    longest = _max_bond_min_image(top, compound.xyz * 10) + 0.5
    start = make_whole(top, compound.xyz * 10) - 0.5 * box
    rng = np.random.default_rng(0)
    path = tmp_path / "traj.gsd"
    with gsd.hoomd.open(str(path), "w") as trajectory:
        for step in range(4):
            # drift every molecule across the boundary, then wrap as HOOMD
            unwrapped = start + step * 0.3 * box + rng.normal(
                0, 0.05, start.shape
            )
            image = np.floor((unwrapped + 0.5 * box) / box).astype(int)
            frame = gsd.hoomd.Frame()
            frame.configuration.step = step * 100
            frame.configuration.box = [*box, 0, 0, 0]
            frame.particles.N = top.n_atoms
            frame.particles.position = unwrapped - image * box
            frame.particles.image = image
            trajectory.append(frame)
    frames, steps = trajectory_positions(top, path)
    assert list(steps) == [0, 100, 200, 300]
    for xyz in frames:
        assert _max_bond(top, xyz) < longest
    # every atom moves by the imposed drift; a wrap would add a box length
    moves = np.linalg.norm(np.diff(frames, axis=0), axis=2)
    drift = np.linalg.norm(0.3 * box)
    assert np.all(np.abs(moves - drift) < 1.0)
    dcd = write_trajectory(top, path, tmp_path / "traj.dcd")
    pdb = write_pdb(top, frames[-1], tmp_path / "top.pdb", whole=False)
    melt = universe(pdb, dcd)
    assert melt.trajectory.n_frames == 4
    melt.trajectory[-1]
    assert np.allclose(melt.atoms.positions, frames[-1], atol=1e-3)


def test_pymol_script(tmp_path):
    from phantomwalk.all_atom.visualization import write_pymol_script

    script = write_pymol_script(
        tmp_path / "melt.pdb", tmp_path / "movie.pml",
        trajectory=tmp_path / "dpd.dcd",
    ).read_text()
    assert "set connect_mode, 1" in script
    assert "load melt.pdb, melt" in script
    assert "load_traj dpd.dcd, melt, state=1" in script


def test_ring_closing_through_the_box_warns():
    from phantomwalk.all_atom.structure import Topology

    box = np.array([9.0, 9.0, 9.0])
    # three atoms a third of the box apart, bonded in a ring: every bond is
    # short by minimum image, but the ring wraps once around the box
    top = Topology(
        names=np.array(["C1", "C2", "C3"], dtype=object),
        elements=np.array(["C", "C", "C"], dtype=object),
        resnames=np.array(["RNG"] * 3, dtype=object),
        resids=np.ones(3, dtype=int),
        segids=np.array(["0000"] * 3, dtype=object),
        molecule=np.zeros(3, dtype=int),
        bonds=np.array([[0, 1], [1, 2], [2, 0]]),
        box_a=box,
    )
    ring = np.array([[0.0, 1, 1], [3.0, 1, 1], [6.0, 1, 1]])
    with pytest.warns(UserWarning, match="periodic image"):
        make_whole(top, ring)
    closed = np.array([[0.0, 1, 1], [1.4, 1, 1], [0.7, 2.2, 1]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_whole(top, closed)
