import numpy as np
import pytest

pytest.importorskip("flowermd")
pytest.importorskip("rdkit")

import mbuild as mb  # noqa: E402

from phantomwalk.all_atom import ion_sites, ion_structure  # noqa: E402


def _acetate_and_sodium(separation_a):
    acetate = mb.load("CC(=O)[O-]", smiles=True)
    sodium = mb.load("[Na+]", smiles=True)
    oxygen = [p for p in acetate.particles() if p.name == "O"][0]
    sodium.translate_to(np.asarray(oxygen.pos) + [separation_a / 10, 0, 0])
    system = mb.Compound([acetate, sodium])
    return system


class TestIonClusters:
    def test_sites(self):
        ions, groups = ion_sites(_acetate_and_sodium(2.3))
        assert len(ions) == 1 and groups.shape == (1, 3)

    def test_contact_and_free(self):
        for separation, free in ((2.3, 0.0), (6.0, 1.0)):
            system = _acetate_and_sodium(separation)
            ions, groups = ion_sites(system)
            s = ion_structure(system.xyz * 10, [50.0] * 3, ions, groups, 3.0)
            assert s["free_ion_fraction"] == free
            assert s["n_aggregates"] == 1

    def test_periodic_contact(self):
        system = _acetate_and_sodium(2.3)
        ions, groups = ion_sites(system)
        xyz = system.xyz * 10
        xyz[ions] += [30.0, 0, 0]  # one box length away
        s = ion_structure(xyz, [30.0] * 3, ions, groups, 3.0)
        assert s["free_ion_fraction"] == 0.0


def test_sage_handoff_small_melt():
    pytest.importorskip("openff.interchange")
    pytest.importorskip("openmm")
    import unyt as u
    from flowermd.library import AllAtomLattice, PolyEthylene

    from phantomwalk.all_atom import sage_handoff

    system = AllAtomLattice(
        PolyEthylene(lengths=10, num_mols=16),
        density=0.5 * u.g / u.cm**3,
        seed=1,
    )
    box_nm = np.asarray(system.system.box.lengths)
    result, positions = sage_handoff(system.system, box_nm, platform="CPU")
    assert result["finite"]
    assert result["energy_removed_sage_epsilon_atom"] > 0
    assert positions.shape == (system.system.n_particles, 3)
    assert result["net_charge_e"] == pytest.approx(0.0, abs=1e-6)
