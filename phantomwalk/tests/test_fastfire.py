"""Tests for the all-atom FastFIRE runner."""

import numpy as np
import pytest

from phantomwalk.lib.fastfire import (
    AllAtomFastFIRESettings,
    _forces,
    _unwrap,
    run_all_atom_fastfire,
)


def test_unwrap_follows_bonds_across_boundary():
    positions = np.array([[4.9, 0, 0], [-4.9, 0, 0], [-4.0, 0, 0]])

    unwrapped = _unwrap(positions, [(0, 1), (1, 2)], np.array([10.0, 10.0, 10.0]))

    np.testing.assert_allclose(unwrapped[:, 0], [4.9, 5.1, 6.0])


def test_short_cpu_fastfire_updates_finite_coordinates():
    mbuild = pytest.importorskip("mbuild")
    compound = mbuild.load("CC", smiles=True)
    compound.translate([1.5, 1.5, 1.5])
    compound.box = mbuild.Box(lengths=[3, 3, 3])

    result = run_all_atom_fastfire(
        compound,
        AllAtomFastFIRESettings(dpd_steps=1, fire_steps=1, device="CPU"),
    )

    assert result.n_particles == 8
    assert result.elapsed_s > 0
    assert np.isfinite(compound.xyz).all()
    assert np.linalg.norm(compound.xyz[0] - compound.xyz[1]) < 0.3


@pytest.mark.parametrize("mode", ("uniform", "class_normalized"))
def test_bonded_scaling_modes_build_complete_force_set(mode):
    mbuild = pytest.importorskip("mbuild")
    hoomd = pytest.importorskip("hoomd")
    from phantomwalk.lib.all_atom import parameterize_all_atom

    compound = mbuild.load("CCCC", smiles=True)
    compound.box = mbuild.Box(lengths=[4, 4, 4])
    parameters = parameterize_all_atom(compound)

    forces = _forces(
        hoomd,
        parameters,
        AllAtomFastFIRESettings(bonded_parameterization=mode),
    )

    assert len(forces) == 4
