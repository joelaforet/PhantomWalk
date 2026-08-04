"""Tests for OpenFF all-atom parameter preparation."""

import numpy as np
import pytest

from phantomwalk.lib.all_atom import parameterize_all_atom


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
