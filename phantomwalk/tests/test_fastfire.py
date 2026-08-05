"""Tests for the all-atom FastFIRE runner."""

import numpy as np
import pytest

from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, _unwrap, run_all_atom_fastfire


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
        AllAtomFastFIRESettings(
            dpd_steps=1,
            fire_steps=1,
            fire_interval=100,
            fire_max_steps=1_000,
            device="CPU",
        ),
    )

    assert result.n_particles == 8
    assert result.elapsed_s > 0
    assert result.fire_converged
    assert result.fire_steps <= 1_000
    assert np.isfinite(compound.xyz).all()
    assert np.linalg.norm(compound.xyz[0] - compound.xyz[1]) < 0.3
