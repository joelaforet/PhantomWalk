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

## All-atom PhantomWalk

The all-atom initializer lives in FlowerMD (`flowermd.library.AllAtomDPD`,
`AllAtomLattice`, `AllAtomPhantomWalk`, currently on the
[`joelaforet/flowerMD`](https://github.com/joelaforet/flowerMD) branch
`aa-dpd-v2/07-smeared-electrostatics`). This repository holds what the
all-atom experiments need around it, in `phantomwalk/all_atom`: the Sage 2.3.0
hand-off minimization and its energy-removed-per-atom metric, and ionomer
ion-aggregate analysis. Demos are in `phantomwalk/examples/all_atom`:

1 - A polyethylene melt from SMILES to Sage-minimized coordinates.
2 - Atactic polystyrene: stereocenters preserved through initialization and minimization.
3 - A sodium ionomer initialized with and without smeared electrostatics, compared by ion aggregates.
4 - The six Colina benchmark polymers (PS, PMMA, PET, PC, PEI, PIM-1), exported and analyzed (density, Rg, end-to-end distance, structure factor).

Every notebook writes its structures to `outputs/<notebook>/`: residue-aware PDB files of
the placement, the initialized (DPD + FIRE) and the Sage-minimized coordinates, the DPD
trajectory as DCD, and PyMOL scripts for pictures and movies. In the PDB files each
monomer is a residue and each molecule a segment (four base-36 characters, so up to
1.68 million molecules), CONECT records hold every bond, and molecules are whole, so they
load directly into MDAnalysis (`phantomwalk.all_atom.universe`), PyMOL and NGLView
(`phantomwalk.all_atom.visualization`).

Set up the environment for these with

``` sh
conda env create -f environment-all-atom.yml
conda activate phantomwalk-all-atom
pip install -e .
python -m ipykernel install --user --name phantomwalk-all-atom
```

and select the `phantomwalk-all-atom` kernel in Jupyter. The coarse-grained workflow above
is unchanged and keeps `environment.yml`.

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
conda activate
```

## Tests

Before running tests, ensure you have activated the [PhantomWalk dev environment](#development-environment).

PhantomWalk tests can be run by invoking `pytest` in the PhantomWalk repo:

``` sh
pytest
```

See the [pytest documentation](https://docs.pytest.org/en/latest/contents.html)
for more details on how `pytest` works and how to use it.
