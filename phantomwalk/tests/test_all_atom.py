"""Tests for OpenFF all-atom parameter preparation."""

import numpy as np
import pytest

from phantomwalk.lib.all_atom import create_openmm_handoff, parameterize_all_atom


def test_sage_230_parameterizes_explicit_hydrogens():
    """Sage labels every ethane bond and supplies positive reduction scales."""

    mbuild = pytest.importorskip("mbuild")
    compound = mbuild.load("CC", smiles=True)
    compound.box = mbuild.Box(lengths=[3, 3, 3])

    parameters = parameterize_all_atom(compound)

    assert len(parameters.positions_a) == 8
    assert len(parameters.bonds) == 7
    assert len(parameters.bond_types) == 7
    assert len(parameters.angles) == 12
    assert parameters.epsilon_ref_kcal_mol > 0
    assert parameters.sigma_ref_a > 0
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
