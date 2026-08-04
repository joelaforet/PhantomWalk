# PhantomWalk
Polymer system initialization workflow that utilizes a random walk and dissipative particle dynamics as a soft push potential.

## Environment

Build a software environment using the `environment.yml` file and the command ```conda env create -f environment.yml```

## Examples
1 - Run a dpd simulation and check the bond lengths and inter-particle distances.
2 - Run a dpd simulation with an energy stabilization cutoff. Write out simulation to trajectory file.
3 - Run a dpd simulation with option for angles and dihedrals. Write out to trajectory file. Start a Lennard-Jones WCA simulation with optional angles and dihedrals.
4 - Replace the random walk in the DPD workflow with mbuild self-avoiding random walk.
5 - Run DPD on a rigid body model used for anisotropic coarse-graining. Based on flowerMD classes.

## All-atom FastFIRE

The all-atom workflow labels an explicit-hydrogen mBuild `Compound` with OpenFF
Sage 2.3.0. It divides energies and lengths by the largest labeled vdW epsilon
and sigma, then runs 500 DPD steps and 200 FIRE steps with the paper defaults:
`A = 250000`, uniform bond `k = 250000`, `gamma = 1500`, and `r_cut = 1.01`.
Sage angles and torsions are retained in reduced units.

``` python
from phantomwalk.lib.all_atom import create_openmm_handoff
from phantomwalk.lib.fastfire import run_all_atom_fastfire

result = run_all_atom_fastfire(melt)
interchange, openmm_system = create_openmm_handoff(melt)
```

The handoff uses Sage's embedded AshGC/NAGL charge model and does not silently
fall back to zero charges. The input must have explicit hydrogens, bonded chains
as top-level children, and periodic `compound.box` lengths.

Run the restartable PE, P3HT, and PES/PSU benchmark matrix with:

``` sh
python -m phantomwalk.benchmarks.aa_test_systems --device GPU \
    --output aa_fastfire_results.jsonl
```

The JSON Lines output separates parameterization, HOOMD setup, DPD, and FIRE
wall times. Existing cases are skipped when the same output file is reused.

## Installation

First, clone the PhantomWalk repository:

``` sh
git clone git@github.com:cmelab/PhantomWalk
cd PhantomWalk
```

Set up & activate the conda environment shipped with PhantomWalk:

``` sh
conda env create -f ./environment.yml
conda activate phantomwalk
```

Install PhantomWalk into the environment with `pip`:

``` sh
python -m pip install -e .
```

### Development Environment

If PhantomWalk is being installed for development purposes, the dev environment
should be used instead, which provides packages necessary for [testing](#tests).

``` sh
conda env create -f ./environment-dev.yml
conda activate phantomwalk-dev
```

## Tests

Before running tests, ensure you have activated the [PhantomWalk dev environment](#development-environment).

PhantomWalk tests can be run by invoking `pytest` in the PhantomWalk repo:

``` sh
pytest
```

See the [pytest documentation](https://docs.pytest.org/en/latest/contents.html)
for more details on how `pytest` works and how to use it.
