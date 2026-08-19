# Status of dense all-atom DPD initialization

Date: 2026-08-19

## Executive status

The current all-atom AA-DPD initializer is the physical-Angstrom protocol in
`phantomwalk/lib/fastfire.py`. The older reduced-unit AA protocol is historical
and should not be described as current. Historical reduced-unit benchmark files
under `benchmark_results/` are archival and are not directly comparable to the
current physical-unit measurements.

The current protocol is empirical but has one workstation validation case at the
final settings. It supports use as an initialization and handoff workflow, not as
an equilibrated-melt generator.

The protocol uses physical Angstrom length, amu mass, and kcal/mol energy
conventions. However, `kT`, `dt`, `gamma`, and the DPD trajectory are empirical
initialization controls, not calibrated physical dynamics.

## Current protocol

1. Build explicit-hydrogen mBuild chains in a target-density periodic box.
2. Create and retain an OpenFF 2.3.0 Interchange with Sage bonded and vdW
   parameters and AshGC/NAGL charges.
3. Run HOOMD DPD/FIRE in physical Angstrom coordinates and force lengths using
   true atomic masses.
4. During DPD/FIRE only, apply OpenFF bonded terms with all coefficients scaled
   by 30. Equilibrium values, periodicities, and phases are retained.
5. Scale each DPD pair by vdW epsilon type:
   `scale_ij = sqrt(epsilon_i * epsilon_j) / epsilon_ref`, where
   `epsilon_ref = max(epsilon_i)`.
6. Copy initialized coordinates into the retained, unscaled OpenFF Interchange.
7. Minimize and optionally run short stability checks in OpenMM CUDA.

Current numerical settings:

| Quantity | Value |
| --- | ---: |
| Base conservative DPD repulsion `A_base` | 5000 |
| Base DPD friction `gamma_base` | 800 |
| DPD cutoff | 3.5 A |
| DPD temperature `kT` | 1 |
| HOOMD timestep `dt` | 0.002 |
| Pair exclusions | 1-2, 1-3, 1-4 (`bond`, `angle`, `dihedral`) |
| DPD baseline | 2000 steps |
| Empirical intensive-energy stopping criterion | 250-step blocks, <=2% energy change |
| Required stopping checks | 2 consecutive checks |
| DPD maximum | 10000 steps |
| FIRE mode | conservative DPD pair force plus bonded terms |
| FIRE minimum | 100 steps |
| FIRE check interval | 200 steps |
| FIRE maximum | 10000 steps |

## Workstation validation case

Hardware and system:

- GPU: NVIDIA RTX 2000 Ada.
- Chemistry: P3HT.
- Size: 10016 atoms, 8 chains x degree 50.
- Initial density: 1.1 g/cm3.
- Random seed: 11.

Measured timings:

| Stage | Time (s) |
| --- | ---: |
| OpenFF Interchange creation | 11.591 |
| GPU DPD, 7500 steps | 14.166 |
| FIRE, 100 steps | 0.090 |
| GPU DPD + FIRE | 14.256 |
| CPU DPD + FIRE from same serialized state | 61.828 |

The same serialized state gave a CPU/GPU DPD+FIRE speedup of 4.337x for this
case. This is a single-system workstation measurement, not a broad scaling
claim.

OpenMM handoff and stability:

| Check | Result |
| --- | ---: |
| OpenMM CUDA initial total potential / (epsilon_ref * atom) | 5.147 |
| OpenMM CUDA final minimized total potential / (epsilon_ref * atom) | 0.821 |
| NVT check | 1000 steps, stable |
| NPT check | 1000 steps, stable |
| Final temperature | 302.9 K |
| Final density | 0.988 g/cm3 |

The 1000-step NVT plus 1000-step NPT trajectory remained finite and did not
explode. This is a short MD stability check only. It does not establish
equilibrium density, production readiness, or thermodynamic sampling quality.
The OpenMM energy normalization is rough: the total potential includes bonded
and electrostatic terms and force-field energy offsets, not only vdW epsilon
interactions.

## Interpretation and limitations

- The final protocol has been validated at one seed and one chemistry at the
  final physical-Angstrom settings.
- The DPD stop is an empirical intensive-energy stopping criterion, not a proof
  of physical convergence or equilibration.
- The PE/P3HT/PES matrix has not yet been repeated at final settings.
- Seed replication has not yet been performed for the final protocol.
- The 2 ps OpenMM check is a handoff stability smoke test, not equilibration.
- FIRE is a deterministic minimizer. It complements stochastic DPD but is not a
  substitute for insufficient DPD overlap relaxation.
- Older reduced-unit runs remain useful negative history, especially for showing
  that literal CG constants and reduced bonded scaling were discarded, but their
  timings and energies should not be mixed with current physical-unit results.

## Current user-facing artifacts

- `phantomwalk/lib/fastfire.py`: final physical-Angstrom AA-DPD/FIRE settings.
- `phantomwalk/lib/all_atom.py`: OpenFF Interchange creation, OpenMM
  minimization, and short dynamics checks.
- `phantomwalk/examples/4-aa-dpd-fastfire.ipynb`: current editable example
  notebook.
- `phantomwalk/benchmarks/README.md`: archival reduced-unit bonded-scaling
  negative result.
