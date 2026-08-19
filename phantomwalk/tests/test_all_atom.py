"""Tests for OpenFF all-atom parameter preparation."""

import numpy as np
import pytest

from phantomwalk.lib.all_atom import (
    create_interchange,
    create_openmm_handoff,
    make_molecules_whole,
    minimize_interchange,
    parameterize_all_atom,
)
from phantomwalk.benchmarks.aa_test_systems import build_chain


def test_make_molecules_whole_centers_each_connected_component():
    positions = np.array(
        [[0.1, 1.0, 1.0], [9.9, 1.0, 1.0], [4.9, -0.1, 2.0], [4.9, 9.8, 2.0]]
    )
    original = positions.copy()

    whole = make_molecules_whole(
        positions, [(0, 1), (2, 3)], np.array([10.0, 10.0, 10.0])
    )

    np.testing.assert_allclose(positions, original)
    np.testing.assert_allclose(
        np.linalg.norm(whole[[1, 3]] - whole[[0, 2]], axis=1), [0.2, 0.1]
    )
    for component in ([0, 1], [2, 3]):
        centroid = whole[component].mean(axis=0)
        assert np.all((centroid >= 0.0) & (centroid < 10.0))


def test_sage_230_parameterizes_explicit_hydrogens():
    """Sage labels every ethane bond and supplies physical parameters."""

    mbuild = pytest.importorskip("mbuild")
    compound = mbuild.load("CC", smiles=True)
    compound.box = mbuild.Box(lengths=[3, 3, 3])

    parameters = parameterize_all_atom(compound)

    assert len(parameters.positions_a) == 8
    assert len(parameters.bonds) == 7
    assert len(parameters.bond_types) == 7
    assert len(parameters.angles) == 12
    assert len(parameters.masses_amu) == 8
    assert len(parameters.atom_types) == 8
    assert parameters.masses_amu.sum() > 30.0
    assert parameters.bond_params
    assert all(values["k"] > 0 and values["r0"] > 0 for values in parameters.bond_params.values())
    assert parameters.epsilon_ref_kcal_mol > 0
    assert set(parameters.atom_types) <= set(parameters.type_epsilons_kcal_mol)
    np.testing.assert_allclose(parameters.box_lengths_a, [30, 30, 30])


def test_parameterizes_rooted_chains_as_separate_molecules():
    """A melt root maps chain-local OpenFF labels to global atom indices."""

    mbuild = pytest.importorskip("mbuild")
    root = mbuild.Compound()
    for shift in (0.5, 2.0):
        chain = mbuild.load("CC", smiles=True)
        chain.translate([shift, 1.5, 1.5])
        root.add(chain)
    root.box = mbuild.Box(lengths=[4, 4, 4])

    parameters = parameterize_all_atom(root)

    assert len(parameters.positions_a) == 16
    assert len(parameters.bonds) == 14
    assert len(parameters.masses_amu) == 16
    assert len(parameters.atom_types) == 16
    assert max(max(group) for group in parameters.bonds) == 15


def test_openmm_handoff_uses_ashgc_charges():
    """Sage's NAGL handler assigns nonzero charges during handoff."""

    mbuild = pytest.importorskip("mbuild")
    compound = mbuild.load("CC", smiles=True)
    compound.translate([1.5, 1.5, 1.5])
    compound.box = mbuild.Box(lengths=[3, 3, 3])

    interchange, system = create_openmm_handoff(compound)
    charges = interchange.collections["Electrostatics"].charges.values()

    assert system.getNumParticles() == 8
    assert any(abs(float(charge.m)) > 1e-6 for charge in charges)


def test_sage_labels_develop_mbuild_polymer_bonds():
    """mBuild path bonds with unspecified order are normalized for OpenFF."""

    pytest.importorskip("mbuild")
    chain = build_chain("pe", degree=4)
    chain.box = pytest.importorskip("mbuild").Box(lengths=[5, 5, 5])

    parameters = parameterize_all_atom(chain)

    assert len(parameters.bonds) == chain.n_bonds


@pytest.mark.parametrize("chemistry", ("p3ht", "pes"))
def test_ashgc_parameterizes_aromatic_test_polymers(chemistry):
    """Tagged aromatic repeat units remain closed-shell after polymerization."""

    mbuild = pytest.importorskip("mbuild")
    degree = 5 if chemistry == "pes" else 4
    chain = build_chain(chemistry, degree=degree)
    chain.box = mbuild.Box(lengths=[8, 8, 8])

    interchange = create_interchange(chain)

    assert len(interchange.collections["Electrostatics"].charges) == chain.n_particles


def test_openmm_minimizes_same_interchange_in_memory():
    """The handoff validation lowers energy and stores minimized coordinates."""

    mbuild = pytest.importorskip("mbuild")
    compound = mbuild.load("CC", smiles=True)
    compound.translate([1.5, 1.5, 1.5])
    compound.box = mbuild.Box(lengths=[3, 3, 3])
    interchange = create_interchange(compound)
    initial_positions = interchange.positions.copy()

    result = minimize_interchange(
        interchange, max_iterations=20, platform_name="CPU"
    )

    assert result.finite
    assert result.minimized_energy_kj_mol <= result.initial_energy_kj_mol
    assert result.platform_name == "CPU"
    assert not np.allclose(interchange.positions, initial_positions)
