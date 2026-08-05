"""Tests for all-atom paper-system construction."""

import pytest

from phantomwalk.benchmarks.aa_test_systems import (
    SYSTEM_DENSITIES,
    build_chain,
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
        line[22:26].strip()
        for line in path.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM"))
    }
    assert len(residue_ids) == 10
