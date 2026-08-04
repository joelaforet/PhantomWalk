"""Build and run the all-atom systems used in the FastFIRE paper benchmarks."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, run_all_atom_fastfire

AMU_NM3_TO_G_CM3 = 1.66053906660e-3
SYSTEM_DENSITIES = {
    "pe": (0.2, 0.8, 1.1),
    "p3ht": (0.2, 0.8, 1.1),
    "pes": (0.2, 0.8, 1.3),
}


def _tagged_monomers(system: str) -> tuple[list[str], str]:
    """Return tagged-SMILES monomers and sequence for a test chemistry."""

    if system == "pe":
        return ["C{<}C{>}"], "A"
    if system == "p3ht":
        return ["c1{<}cc(sc1{>}CCCCCC)"], "A"
    if system == "pes":
        bisphenol_a = "c1{<}ccc(C(C)(C)c2ccc({>}cc2))cc1"
        diphenyl_sulfone = "O=S(=O)(c1{<}ccc(cc1))c1ccc({>}cc1)"
        # Three BPA and two BPS units give the requested 40:60 BPS:BPA ratio.
        return [bisphenol_a, diphenyl_sulfone], "AABAB"
    raise ValueError(f"unknown system {system!r}; choose from {tuple(SYSTEM_DENSITIES)}")


def build_chain(system: str, degree: int = 50) -> Any:
    """Build one explicit-hydrogen chain with mBuild's develop path API."""

    import mbuild as mb
    from mbuild.path import straight_line

    smiles, sequence = _tagged_monomers(system)
    polymer = mb.Polymer()
    for monomer_smiles in smiles:
        monomer = mb.load(monomer_smiles, smiles=True)
        polymer.add_monomer(monomer, head_tag="<", tail_tag=">", separation=0.15)
    path = straight_line(N=degree * len(sequence), spacing=0.5)
    polymer.build_from_path(path, sequence=sequence, energy_minimize=False)
    return polymer


def build_test_system(
    system: str,
    density_g_cm3: float,
    target_atoms: int,
    degree: int = 50,
    seed: int = 11,
) -> Any:
    """Clone chains into a cubic box sized to the requested mass density."""

    import mbuild as mb

    chain = build_chain(system, degree)
    n_chains = max(1, int(np.ceil(target_atoms / chain.n_particles)))
    root = mb.Compound(name=system.upper())
    rng = np.random.default_rng(seed)
    chain_mass = sum(float(particle.mass) for particle in chain.particles())
    box_length = (n_chains * chain_mass * AMU_NM3_TO_G_CM3 / density_g_cm3) ** (1 / 3)
    for _ in range(n_chains):
        copy = mb.clone(chain)
        copy.translate(rng.uniform(0, box_length, size=3) - copy.center)
        root.add(copy)
    root.box = mb.Box(lengths=[box_length] * 3)
    return root


def _completed_keys(path: Path) -> set[tuple[Any, ...]]:
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text().splitlines():
        row = json.loads(line)
        keys.add((row["system"], row["density_g_cm3"], row["target_atoms"], row["seed"]))
    return keys


def run_matrix(output: Path, device: str = "auto") -> None:
    """Run the restartable three-chemistry benchmark matrix as JSON Lines."""

    completed = _completed_keys(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for system, densities in SYSTEM_DENSITIES.items():
        for density in densities:
            for target_atoms in (10_000, 20_000, 40_000, 80_000):
                for seed in (11, 22, 33, 44, 55):
                    key = (system, density, target_atoms, seed)
                    if key in completed:
                        continue
                    compound = build_test_system(system, density, target_atoms, seed=seed)
                    result = run_all_atom_fastfire(
                        compound,
                        AllAtomFastFIRESettings(seed=seed, device=device),
                    )
                    row = dict(
                        system=system,
                        density_g_cm3=density,
                        target_atoms=target_atoms,
                        actual_atoms=compound.n_particles,
                        seed=seed,
                        **asdict(result),
                    )
                    with output.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("aa_fastfire_results.jsonl"))
    parser.add_argument("--device", choices=("auto", "CPU", "GPU"), default="auto")
    args = parser.parse_args()
    run_matrix(args.output, args.device)


if __name__ == "__main__":
    main()
