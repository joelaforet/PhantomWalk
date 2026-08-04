"""Tests for all-atom paper-system construction."""

import pytest

from phantomwalk.benchmarks.aa_test_systems import SYSTEM_DENSITIES, build_chain


@pytest.mark.parametrize("system", SYSTEM_DENSITIES)
def test_build_short_chain(system):
    pytest.importorskip("mbuild")

    chain = build_chain(system, degree=10)

    assert chain.n_particles > 0
    assert chain.n_bonds == chain.n_particles - 1 or system != "pe"
