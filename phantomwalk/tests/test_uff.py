"""Focused correctness tests for the order-preserving RDKit UFF provider."""

import math
import numpy as np
import pytest

from phantomwalk.lib.uff import parameterize_uff, uff_energy_components


def _compound(smiles="CC"):
    mb = pytest.importorskip("mbuild")
    compound = mb.load(smiles, smiles=True)
    compound.box = mb.Box(lengths=[4, 4, 4])
    return compound


def test_uff_ethane_coverage_units_order_and_determinism():
    first = parameterize_uff(_compound())
    second = parameterize_uff(_compound())

    assert len(first.positions_a) == 8
    assert (len(first.bonds), len(first.angles), len(first.dihedrals), len(first.impropers)) == (7, 12, 9, 0)
    np.testing.assert_allclose(first.box_lengths_a, [40, 40, 40])
    np.testing.assert_allclose(first.masses_amu[:2], [12.011, 12.011])
    assert first.bonds == second.bonds
    assert first.angles == second.angles
    assert first.dihedrals == second.dihedrals
    assert first.bond_params == second.bond_params
    assert first.angle_params == second.angle_params
    assert first.dihedral_params == second.dihedral_params
    assert first.epsilon_ref_kcal_mol == max(v["epsilon_kcal_mol"] for v in first.particle_type_params.values())


def test_coefficients_equal_rdkit_getters():
    from rdkit import Chem
    from rdkit.Chem import rdForceFieldHelpers as uff

    compound = _compound("c1ccsc1")
    mol = compound.to_rdkit()
    Chem.SanitizeMol(mol)
    parameters = parameterize_uff(compound)
    for group, name in zip(parameters.bonds, parameters.bond_types):
        kb, r0 = uff.GetUFFBondStretchParams(mol, *group)
        assert parameters.bond_params[name] == pytest.approx({"k": kb, "r0": r0})
    for group, name in zip(parameters.angles, parameters.angle_types):
        ka, theta0 = uff.GetUFFAngleBendParams(mol, *group)
        assert parameters.angle_params[name]["k"] == pytest.approx(ka)
        assert parameters.angle_params[name]["t0"] == pytest.approx(math.radians(theta0))
    counts = {}
    for group in parameters.dihedrals:
        counts[group[1:3]] = counts.get(group[1:3], 0) + 1
    for group, name in zip(parameters.dihedrals, parameters.dihedral_types):
        expected = uff.GetUFFTorsionParams(mol, *group) / counts[group[1:3]]
        assert parameters.dihedral_params[name]["k"] == pytest.approx(expected)
    for group, name in zip(parameters.impropers, parameters.improper_types):
        assert parameters.improper_params[name]["k"] == pytest.approx(uff.GetUFFInversionParams(mol, *group))


def test_reference_uff_bonded_energies_are_finite_when_perturbed():
    parameters = parameterize_uff(_compound("c1ccsc1"))
    at_input = uff_energy_components(parameters, parameters.positions_a)
    perturbed = parameters.positions_a.copy()
    perturbed[0] += [0.07, -0.04, 0.03]
    at_perturbed = uff_energy_components(parameters, perturbed)
    assert all(np.isfinite(list(at_input.values())))
    assert all(np.isfinite(list(at_perturbed.values())))
    assert at_input != at_perturbed


def test_analytical_bonded_energy_matches_rdkit_at_input_and_perturbation():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    compound = _compound("CC")
    parameters = parameterize_uff(compound)
    mol = compound.to_rdkit()
    Chem.SanitizeMol(mol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    mol.AddConformer(conformer)
    conformer = mol.GetConformer()
    for displacement in (np.zeros(3), np.array([0.07, -0.04, 0.03])):
        positions = parameters.positions_a.copy()
        positions[0] += displacement
        for index, position in enumerate(positions):
            conformer.SetAtomPosition(index, position)
        expected = AllChem.UFFGetMoleculeForceField(mol).CalcEnergy()
        observed = sum(uff_energy_components(parameters, positions).values())
        assert observed == pytest.approx(expected, rel=2e-11, abs=2e-11)


def test_atom_order_mismatch_fails_clearly(monkeypatch):
    from rdkit import Chem

    compound = _compound()
    original = compound.to_rdkit()
    order = list(range(original.GetNumAtoms()))
    order[0], order[2] = order[2], order[0]
    monkeypatch.setattr(compound, "to_rdkit", lambda: Chem.RenumberAtoms(original, order))
    with pytest.raises(ValueError, match="atom-order mismatch"):
        parameterize_uff(compound)


def test_repeated_disconnected_chains_are_merged_with_global_indices():
    mb = pytest.importorskip("mbuild")
    root = mb.Compound()
    root.add([_compound("CC"), _compound("CC")])
    root.box = mb.Box(lengths=[8, 8, 8])

    parameters = parameterize_uff(root)

    assert len(parameters.positions_a) == 16
    assert len(parameters.bonds) == 14
    assert len(parameters.masses_amu) == 16
    assert max(max(group) for group in parameters.bonds) == 15


def test_distinct_disconnected_chemistries_remap_local_type_names():
    mb = pytest.importorskip("mbuild")
    root = mb.Compound()
    root.add([_compound("CC"), _compound("c1ccsc1")])
    root.box = mb.Box(lengths=[8, 8, 8])

    parameters = parameterize_uff(root)

    assert len(parameters.positions_a) == root.n_particles
    assert len(parameters.particle_types) == root.n_particles
    assert set(parameters.particle_types) <= set(parameters.particle_type_params)
    assert set(parameters.bond_types) <= set(parameters.bond_params)
