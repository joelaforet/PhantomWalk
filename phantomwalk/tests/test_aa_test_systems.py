"""Tests for all-atom paper-system construction."""

import numpy as np
import pytest

from phantomwalk.benchmarks.aa_test_systems import (
    AMU_NM3_TO_G_CM3,
    SYSTEM_DENSITIES,
    _base36,
    build_chain,
    build_melt,
    build_test_system,
    shortest_periodic_one_four_distance_a,
    write_visualization_pdb,
)


@pytest.mark.parametrize("system", SYSTEM_DENSITIES)
def test_build_short_chain(system):
    pytest.importorskip("mbuild")

    chain = build_chain(system, degree=10)

    assert chain.n_particles > 0
    assert chain.n_bonds == chain.n_particles - 1 or system != "pe"


def test_visualization_pdb_assigns_one_residue_per_monomer(tmp_path):
    mbuild = pytest.importorskip("mbuild")
    chain = build_chain("pe", degree=10)
    chain.box = mbuild.Box(lengths=[10, 10, 10])
    path = tmp_path / "pe.pdb"

    write_visualization_pdb(chain, path)

    residue_ids = {
        (line[72:76].strip(), line[22:26].strip())
        for line in path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    }
    assert len(residue_ids) == 10


def test_visualization_pdb_assigns_unique_polymer_segments(tmp_path):
    pytest.importorskip("mbuild")
    melt = build_test_system("pe", density_g_cm3=0.2, target_atoms=1_000, degree=10)
    path = tmp_path / "melt.pdb"

    write_visualization_pdb(melt, path)

    segments = {
        line[72:76].strip()
        for line in path.read_text().splitlines()
        if line.startswith("HETATM")
    }
    assert len(segments) == len(melt.children)
    assert _base36(35) == "Z"
    assert _base36(36) == "10"


def test_visualization_pdb_writes_whole_centered_molecules_without_mutation(
    tmp_path,
):
    mbuild = pytest.importorskip("mbuild")
    molecule = mbuild.Compound(name="Polymer")
    first = mbuild.Compound(name="C", element="C", pos=[0.1, 1.5, 1.5])
    second = mbuild.Compound(name="C", element="C", pos=[2.9, 1.5, 1.5])
    molecule.add([first, second])
    molecule.add_bond((first, second))
    molecule.box = mbuild.Box(lengths=[3.0, 3.0, 3.0])
    original = molecule.xyz.copy()
    path = tmp_path / "whole.pdb"

    write_visualization_pdb(molecule, path)

    coordinates = np.asarray(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in path.read_text().splitlines()
            if line.startswith("HETATM")
        ]
    )
    np.testing.assert_allclose(molecule.xyz, original)
    assert np.linalg.norm(coordinates[1] - coordinates[0]) == pytest.approx(2.0)
    assert np.all((coordinates.mean(axis=0) >= 0.0) & (coordinates.mean(axis=0) < 30.0))


def test_shortest_periodic_one_four_distance_uses_minimum_image():
    mbuild = pytest.importorskip("mbuild")
    molecule = mbuild.Compound()
    particles = [
        mbuild.Compound(name="C", element="C", pos=position)
        for position in ([0.1, 1, 1], [0.2, 1, 1], [0.3, 1, 1], [2.9, 1, 1])
    ]
    molecule.add(particles)
    for first, second in zip(particles, particles[1:]):
        molecule.add_bond((first, second))
    molecule.box = mbuild.Box(lengths=[3, 3, 3])

    assert shortest_periodic_one_four_distance_a(molecule) == pytest.approx(2.0)


def test_build_melt_uses_requested_chain_count_and_density():
    pytest.importorskip("mbuild")

    melt = build_melt("pe", n_chains=3, degree=10, density_g_cm3=0.8)
    mass = sum(float(particle.mass) for particle in melt.particles())
    volume = float(melt.box.lengths[0] ** 3)

    assert len(melt.children) == 3
    assert mass * AMU_NM3_TO_G_CM3 / volume == pytest.approx(0.8)


def test_build_melt_default_density_is_highest_for_chemistry():
    pytest.importorskip("mbuild")

    melt = build_melt("pe", n_chains=1, degree=10)
    mass = sum(float(particle.mass) for particle in melt.particles())
    volume = float(melt.box.lengths[0] ** 3)

    assert mass * AMU_NM3_TO_G_CM3 / volume == pytest.approx(max(SYSTEM_DENSITIES["pe"]))


def test_build_melt_rejects_nonpositive_chain_count():
    with pytest.raises(ValueError):
        build_melt("pe", n_chains=0)
