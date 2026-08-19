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

The current all-atom AA-DPD initializer operates in physical Angstrom
coordinates and force lengths. Explicit-hydrogen mBuild chains are labeled with
OpenFF Sage 2.3.0, true atomic masses are retained in HOOMD, and OpenFF bond,
angle, proper-torsion, and improper-torsion equilibrium values, periodicities,
and phases are retained. During DPD/FIRE only, all OpenFF bonded force
coefficients are multiplied by 30. The unscaled OpenFF Interchange is retained
and restored for the OpenMM handoff. Although the initializer uses physical
Angstrom, amu, and kcal/mol conventions, `kT`, `dt`, `gamma`, and the DPD
trajectory are empirical initialization controls, not calibrated physical
dynamics.

The DPD pair strength is type-dependent:
`A_ij = A_base * sqrt(epsilon_i * epsilon_j) / epsilon_ref`, with
`A_base = 5000` and `epsilon_ref = max(epsilon_i)` from the Sage vdW labels.
Friction is scaled the same way from `gamma_base = 800`. The current settings
are `r_cut = 3.5 A`, `kT = 1`, `dt = 0.002`, and 1-2/1-3/1-4 exclusions
(`bond`, `angle`, `dihedral`). DPD runs for 2,000 baseline steps, then applies
an empirical intensive-energy stopping criterion: 250-step checks continue until
all intensive bonded and pair energies change by at most 2% for two consecutive
checks, capped at 10,000 steps. Conservative FIRE then runs for at least 100
steps, checking every 200 steps up to 10,000 steps.

The older reduced-unit AA protocol and literal coarse-grained constants are
discarded for current use. See
[`benchmark_results/AA_DPD_STATUS_REPORT.md`](benchmark_results/AA_DPD_STATUS_REPORT.md)
for workstation validation evidence and limitations.

``` python
from phantomwalk.benchmarks.aa_test_systems import build_melt
from phantomwalk.lib.all_atom import (
    create_interchange,
    minimize_interchange,
    run_interchange_dynamics,
)
from phantomwalk.lib.fastfire import AllAtomFastFIRESettings, run_all_atom_fastfire

melt = build_melt("p3ht", n_chains=8, degree=50, density_g_cm3=1.1, seed=11)
interchange = create_interchange(melt)
fastfire = run_all_atom_fastfire(
    melt,
    AllAtomFastFIRESettings(device="GPU", seed=11),
    interchange=interchange,
)
minimization = minimize_interchange(interchange)
dynamics = run_interchange_dynamics(interchange, nvt_steps=1000, npt_steps=1000)
```

`build_melt` is configurable by chemistry (`"pe"`, `"p3ht"`, or `"pes"`),
`n_chains`, polymer `degree`, `density_g_cm3`, and `seed`. The handoff uses
Sage's embedded AshGC/NAGL charge model and does not silently fall back to zero
charges. The input must have explicit hydrogens, bonded chains as top-level
children, and periodic `compound.box` lengths.

The example notebook
[`phantomwalk/examples/4-aa-dpd-fastfire.ipynb`](phantomwalk/examples/4-aa-dpd-fastfire.ipynb)
runs the current physical-Angstrom AA-DPD/FastFIRE workflow. It exposes editable
chemistry, chain count, degree, density, seed, device, and short OpenMM MD
settings. The short NVT/NPT section is a stability smoke test, not an
equilibration or production protocol.

For GPU AA-DPD work, use the dedicated environment and register its notebook
kernel:

``` sh
conda env create -f environment-gpu.yml
conda activate phantomwalk-gpu
python -m pip install --no-deps -e .
python -m ipykernel install --user --name phantomwalk-gpu --display-name "PhantomWalk GPU"
```

The portable `environment.yml` and `environment-dev.yml` allow conda to select
a CPU HOOMD build. The dedicated GPU file requires a GPU-enabled HOOMD build so
the showcase cannot silently start with a CPU-only package.

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
