"""Tests for the all-atom FastFIRE runner."""

import numpy as np
import pytest

from phantomwalk.lib.fastfire import (
    AllAtomFastFIRESettings,
    _epsilon_per_atom_energies,
    _energies_converged,
    _intensive_energies,
    _unwrap,
    run_all_atom_fastfire,
)


def test_unwrap_follows_bonds_across_boundary():
    positions = np.array([[4.9, 0, 0], [-4.9, 0, 0], [-4.0, 0, 0]])

    unwrapped = _unwrap(positions, [(0, 1), (1, 2)], np.array([10.0, 10.0, 10.0]))

    np.testing.assert_allclose(unwrapped[:, 0], [4.9, 5.1, 6.0])


def test_intensive_energy_convergence_is_size_independent():
    counts = {"bond": 10, "angle": 8}
    first = _intensive_energies(
        {"bond": 100.0, "angle": 40.0, "pair": 200.0}, 20, counts
    )
    second = _intensive_energies(
        {"bond": 101.0, "angle": 40.4, "pair": 202.0}, 20, counts
    )

    assert first == {
        "bond_energy_per_interaction": 10.0,
        "angle_energy_per_interaction": 5.0,
        "pair_energy_per_atom": 10.0,
    }
    assert _energies_converged(first, second, 0.02)
    assert not _energies_converged(first, {**second, "pair_energy_per_atom": 10.3}, 0.02)


def test_epsilon_per_atom_energies_match_cg_normalization():
    normalized = _epsilon_per_atom_energies(
        {"bond": 80.0, "pair": 120.0}, n_particles=20, epsilon_ref=2.0
    )

    assert normalized == {
        "bond_energy_epsilon_per_atom": 2.0,
        "pair_energy_epsilon_per_atom": 3.0,
        "total_energy_epsilon_per_atom": 5.0,
    }


def test_short_cpu_fastfire_updates_finite_coordinates():
    mbuild = pytest.importorskip("mbuild")
    compound = mbuild.load("CC", smiles=True)
    compound.translate([1.5, 1.5, 1.5])
    compound.box = mbuild.Box(lengths=[3, 3, 3])

    result = run_all_atom_fastfire(
        compound,
        AllAtomFastFIRESettings(
            dpd_steps=1,
            dpd_max_steps=1,
            require_dpd_convergence=False,
            fire_steps=1,
            fire_interval=100,
            fire_max_steps=1_000,
            require_fire_convergence=False,
            device="CPU",
        ),
    )

    assert result.n_particles == 8
    assert result.elapsed_s > 0
    assert result.fire_steps <= 1_000
    assert result.device_description
    assert result.epsilon_ref_kcal_mol > 0
    assert result.dpd_steps == 1
    assert not result.dpd_converged
    assert np.isfinite(compound.xyz).all()
    assert np.linalg.norm(compound.xyz[0] - compound.xyz[1]) < 0.3
