# Status of dense all-atom DPD initialization

Date: 2026-08-05

## Executive conclusion

The literal numerical protocol developed for Eric's dense linear coarse-grained
(CG) bead-spring chains does **not** transfer to explicit-hydrogen all-atom
(AA) melts. In particular, `A = 250000`, `gamma = 1500`, a 500-step DPD phase,
and assigning reduced stiffnesses of 250,000 broadly across bonded terms caused
collapsed local geometry, numerical instability, or extremely unfavorable
production-force-field handoffs.

The underlying physical strategy does transfer: construct overlapping chains
quickly with periodic random walks, use a soft DPD potential to remove those
overlaps, use FIRE for inexpensive deterministic cleanup, and then activate the
production force field. The AA implementation should therefore be described as
an **all-atom adaptation of the same initialization principle**, but it is a
distinct numerical protocol with topology-aware bonded forces, a weaker DPD
repulsion, a smaller timestep, and convergence-based rather than fixed-duration
relaxation.

This distinction is expected. Eric's model has one interaction site per CG
bead, simple linear connectivity, and low local topological complexity. The AA
systems contain explicit hydrogens, rings, side chains, chemically distinct
angles and torsions, impropers, and many more interacting sites per physical
volume. P3HT contains a branched side chain but is not a truly branched polymer
backbone; a genuinely branched AA topology remains to be tested.

## What the current workflow does

1. Build explicit-hydrogen chains with the mBuild development random-walk API
   inside a periodic volume constraint.
2. Clone independently generated chains directly into the target-density box.
   Chains may overlap and pass through one another. Packmol is not used.
3. Create and retain an OpenFF Interchange using Sage 2.3.0 and AshGC charges.
4. Extract Sage bonds, angles, proper torsions, and impropers and apply them in
   HOOMD during DPD and FIRE.
5. Run stochastic DPD until intensive bonded and pair energies stabilize.
6. Replace stochastic DPD with its conservative pair force and run FIRE.
7. Copy the HOOMD coordinates directly into the retained Interchange.
8. Export to OpenMM, minimize without an iteration cap, and optionally run short
   production-force-field NVT and NPT checks.

The random walk currently uses a very small hard-sphere radius of 0.05 nm and a
0.30 nm path bond length. It therefore avoids only the most degenerate
intrachain construction events; it is not a packed or globally self-avoiding
melt. Independently built chains are allowed to overlap, as intended.

## Current AA-DPD parameters

### Reduced units

For each chemistry:

- Energy reference: the largest labeled Sage vdW epsilon,
  `epsilon_ref = max(epsilon_i)`.
- Length reference: the largest labeled Sage vdW sigma,
  `sigma_ref = max(sigma_i)`.
- Coordinates and equilibrium bond lengths are divided by `sigma_ref`.
- Sage angular and torsional energies are divided by `epsilon_ref`.

| Chemistry | epsilon_ref (kcal/mol) | sigma_ref (A) | High-density state (g/cm3) | Reduced atom number density |
|---|---:|---:|---:|---:|
| PE | 0.12057 | 3.38797 | 1.1 | 5.54 |
| P3HT | 0.25000 | 3.56359 | 1.1 | 4.51 |
| PES | 0.25000 | 3.56359 | 1.3 | 4.44 |

The reduced number density is

`rho* = N * sigma_ref^3 / V`.

These values are far above Eric's CG regime, where performance and geometry
were favorable below approximately `rho* = 1.2`. This is a major reason a CG
repulsion of 250,000 cannot be transferred literally: each AA site has many
more neighbors inside the reduced cutoff, so the accumulated DPD force on
local bonded structures is much larger.

### DPD and bonded forces

| Parameter | Current value | Interpretation |
|---|---:|---|
| DPD repulsion `A` | 25,000 | One reduced pair type for every atom |
| DPD friction `gamma` | 800 | Stochastic DPD phase only |
| DPD temperature `kT` | 1.0 | Reduced units |
| DPD cutoff | 1.01 | Reduced by `sigma_ref` |
| HOOMD timestep | 0.0001 | Ten times smaller than Eric's nominal CG timestep |
| Bond stiffness | 250,000 | Uniform reduced harmonic `k`; Sage equilibrium lengths retained |
| Angle strength | `30 * k_OpenFF / epsilon_ref` | Sage relative parameterization retained |
| Proper strength | `30 * k_OpenFF / epsilon_ref` | Sage periodicity and phase retained |
| Improper strength | `30 * k_OpenFF / epsilon_ref` | Sage periodicity and phase retained |
| Pair exclusions | bond, angle, dihedral | Removes DPD 1-2, 1-3, and 1-4 competition |

The 30-fold bonded scaling follows the working MuPT-style AA implementation.
It is not claimed to reproduce the production potential during DPD. Its purpose
is to preserve chemically reasonable local geometry while the soft pair force
removes interchain and long-range intrachain overlaps.

### Adaptive duration

- DPD baseline: 2,000 steps.
- DPD extension: 250 steps.
- DPD maximum: 10,000 steps.
- DPD convergence: every pair/bond/angle/proper/improper energy, normalized per
  particle or per interaction, must change by at most 2% for two consecutive
  extension windows.
- FIRE minimum: 100 steps.
- FIRE extension interval: 200 steps.
- FIRE maximum: 10,000 steps.
- FIRE tolerances: force and energy tolerances of 1,000 in HOOMD reduced units.

DPD uses the stochastic dissipative force. FIRE is not DPD: it is a
deterministic energy minimizer. During FIRE the stochastic force is removed and
only the conservative DPD repulsion plus bonded forces remain.

## Experiments that did not work

### Literal CG-like repulsion

Using `A = 250000` in AA systems overwhelmed the bonded geometry. Representative
minimum bond/1-3/1-4 distances after the AA relaxation were approximately:

- P3HT: 0.26/0.19/0.20 A.
- PES: 0.21/0.13/0.06 A.

The resulting OpenMM energies were dominated by the production nonbonded force
and reached approximately 10^12--10^18 kJ/mol in tested systems. OpenMM could
sometimes numerically rescue a small system, but this is not a defensible or
size-robust handoff.

Reducing `A` to 5,000 preserved local bonds better, with minimum bonds near
0.93 A, but left severe nonexcluded/interchain contacts of roughly 0.37--0.55
A. The selected `A = 25000` is the tested compromise between those failure
modes.

### Mapping every bonded class to 250,000

Two alternatives were tested near 10,000 atoms:

1. Assign every bond, angle, proper, and improper a reduced `k` of 250,000.
2. Preserve relative OpenFF strengths within each class but normalize the
   largest member of every class to 250,000.

Negative results included:

- PE: both policies became unstable during DPD, after 21.27 and 13.93 s.
- P3HT uniform: unstable during DPD after 28.02 s.
- P3HT class-normalized: completed, but handed OpenMM a structure at
  `1.72e20 kJ/mol`; pre-minimization bond RMS error was 1.54 A.
- PES uniform: DPD/FIRE completed, but the minimization experiment was stopped
  after the preceding failures rejected the policy.

Those experimental code paths were reverted to avoid repository bloat. The
negative result is retained in `phantomwalk/benchmarks/README.md`.

### Too little DPD followed by more FIRE

The 2k-atom iteration sweep initially made 250 DPD + 400 FIRE look attractive:
it passed PE, P3HT, and PES for three small-system seeds and was faster than
1000 DPD. At approximately 10k atoms it failed for two of three chemistries:

- P3HT retained a 0.587 A nonexcluded contact.
- PES developed a 0.243 A 1-4 distance and an initial OpenMM energy of
  `3.25e7 kJ/mol/atom`.

This establishes that FIRE cannot reliably substitute for missing stochastic
DPD relaxation. DPD removes the random-walk overlap field; FIRE efficiently
cleans up only after that field has been sufficiently relaxed.

### Fixed 1000 DPD + 50 FIRE

This setting passed the three approximately 10k systems, but P3HT remained
close to the local-geometry threshold and entered OpenMM at
`7.97e6 kJ/mol/atom`. Increasing FIRE to 100 steps cost essentially no measured
wall time and improved that P3HT handoff approximately ninefold, to
`8.86e5 kJ/mol/atom`, but the value was still much higher than desired.

The fixed 1000/100 protocol is the fastest tested protocol that passed the 10k
screen, but it is no longer the recommended robust protocol.

## Positive results from the adaptive AA protocol

### Scaling and geometry

| Chemistry | Atoms | DPD steps | DPD time (s) | Closest nonexcluded (A) | OpenMM initial (kJ/mol/atom) | OpenMM final (kJ/mol/atom) |
|---|---:|---:|---:|---:|---:|---:|
| PE | 10,268 | 3,500 | 63.18 | 1.89 | 9.72 | 3.23 |
| PE | 40,166 | 3,500 | 263.12 | 1.81 | 11.03 | 4.11 |
| PE | 80,030 | 3,500 | 443.07 | 1.79 | 11.53 | 4.33 |
| P3HT | 10,016 | 4,750 | 67.68 | 2.02 | 78.33 | 1.27 |
| P3HT | 40,064 | 4,500 | 274.73 | 2.03 | 36.26 | 1.21 |
| PES | 11,136 | 5,250 | 89.28 | 2.17 | 28.73 | 1.63 |
| PES | 40,368 | 5,750 | 267.14 | 2.05 | 28.80 | 1.58 |

All of these structures minimized to finite energies with uncapped OpenMM
minimization on CUDA. The OpenMM minimizer used Sage 2.3.0 with AshGC charges,
a force tolerance of 10 kJ/mol/nm, and `maxIterations = 0`.

The adaptive iteration count is chemistry-dependent but did not increase
systematically with atom count. PE used 3,500 DPD steps from 10k through 80k;
P3HT used 4,500--4,750 from 10k to 40k; PES used 5,250--5,750. At fixed density,
the data support roughly linear wall time per DPD step, not DPD iteration count
proportional to atom count.

### Performance relative to Eric's CG benchmark

Eric reported roughly 1.6--2.1 s for 10,000 CG particles and approximately
0.0002 s/particle for 500 DPD plus 200 FIRE steps below reduced number density
1.2. The adaptive AA 10k DPD phases required 63--89 s, or roughly
0.006--0.008 s/atom, before adding construction, Interchange creation, FIRE,
diagnostics, and OpenMM.

This is not a controlled CPU/GPU comparison: Eric's figures are GPU results,
whereas HOOMD AA-DPD ran on the CPU because the local HOOMD/GPU path was not
reliable. The AA protocol also runs 3,500--5,250 DPD steps instead of 500 and
carries several bonded force classes. It is therefore fair to conclude that we
have not reproduced the CG seconds-per-particle constant for AA, but not fair to
attribute the entire difference to the AA algorithm rather than device and step
count. Within the AA data, DPD wall time remains approximately linear in atom
count.

### Production-force-field dynamics

After minimization, short production checks used:

- OpenMM CUDA mixed precision.
- Langevin-middle integration at 300 K.
- Friction of 1/ps.
- Timestep of 2 fs.
- 1,000 NVT steps followed by 1,000 NPT steps.
- Monte Carlo barostat at 1 bar, attempted every 25 steps.

| Chemistry | Atoms | Finite | Final NPT T (K) | Final density (g/cm3) | Sampled potential-energy range (kJ/mol/atom) |
|---|---:|---:|---:|---:|---:|
| PE | 2,114 | yes | 298.0 | 1.019 | 4.51--5.58 |
| P3HT | 2,504 | yes | 306.3 | 0.998 | 3.15--3.85 |
| PES | 2,784 | yes | 302.6 | 1.165 | 3.63--4.60 |
| PE | 80,030 | yes | 312.3 | 1.008 | 5.20--6.52 |

No tested NVT or NPT trajectory exploded. Energies stayed finite and in narrow
bands. The density changes were smooth and moved away from the deliberately
high initial densities. These 2 ps smoke tests demonstrate handoff stability;
they are not long enough to establish equilibrium density or thermodynamic
production readiness.

## Answers to Eric's questions

### How much work remains for OpenMM minimization?

Current PhantomWalk logs record endpoint energies rather than every L-BFGS
iteration. OpenMM 8.5 does expose a `MinimizationReporter` callback that can
record energy and force information after each iteration; the current wrapper
does not attach one. Eric's requested minimization trace can therefore be added
without repeatedly restarting the minimizer. Coordinate RMSD or displacement
during minimization has not yet been measured.

Using the largest Sage epsilon as Eric's requested reference:

- PE 40k: 11.03 to 4.11 kJ/mol/atom. With
  `epsilon_ref = 0.12057 kcal/mol = 0.504 kJ/mol`, OpenMM removes approximately
  13.7 epsilon per atom.
- P3HT 40k: 36.26 to 1.21 kJ/mol/atom. With
  `epsilon_ref = 0.25 kcal/mol = 1.046 kJ/mol`, OpenMM removes approximately
  33.5 epsilon per atom.
- PES 40k: 28.80 to 1.58 kJ/mol/atom, or approximately 26.0 epsilon per atom.

Thus OpenMM is doing meaningful cleanup in the tens of epsilon per particle,
not rescuing thousands of epsilon per particle under the adaptive protocol.
Absolute AA energy divided by one LJ epsilon is only an approximate comparison:
the production energy also contains electrostatics and chemically heterogeneous
bonded terms with a different energy zero.

The production cutoff being longer than the DPD cutoff does not mean all newly
included pairs become strongly repulsive. Most added LJ pairs lie in the weak
or attractive region, and electrostatics are also activated. Catastrophic
handoffs are driven primarily by rare short-range contacts, which are tested
with nonexcluded distance and initial production energy.

### Does production NVT or NPT explode?

No explosion was observed in the tested 2 fs smoke tests for PE, P3HT, or PES,
including an 80,030-atom PE system. This is positive handoff evidence, but it
does not replace longer equilibration, multiple seeds, pressure/density
statistics, or production-property validation.

### Is FIRE different from DPD, and can it run to convergence?

Yes. DPD is stochastic dynamics with conservative, dissipative, and random
pair forces. FIRE is a deterministic minimizer. HOOMD FIRE exposes a convergence
flag, so the implementation runs at least 100 steps and can continue in
200-step increments to a 10,000-step cap. The experiments show that converged
FIRE is not sufficient when the preceding DPD phase is too short.

### Did DPD/FIRE and OpenMM use the GPU?

HOOMD DPD/FIRE used the CPU because HOOMD did not operate reliably on the local
GPU. OpenMM minimization, NVT, and NPT used the CUDA platform in mixed precision
on the RTX 3080 Ti Laptop GPU.

## Negative or incomplete evidence

- The adaptive convergence rule is empirical. A 2% change in every intensive
  energy for two windows is supported by current tests but is not a theorem
  guaranteeing removal of every possible rare clash.
- Large-system results currently use one seed. The 2k finalist study used three
  seeds, but the adaptive 40k and 80k cases need seed replication.
- Only PE has completed the full 80k initialization/minimization/NVT/NPT chain.
- The NVT/NPT checks total only 2 ps and do not establish equilibrium.
- A true branched-backbone AA polymer has not been tested.
- P3HT and PES remain more expensive in required DPD steps than PE.
- The 800k case has not been run. Measured PE host-memory scaling projects
  approximately 29.4 GiB for the current mBuild/OpenFF/Interchange workflow,
  exceeding the workstation's 16 GiB. A 48--64 GiB host is recommended.
- Legacy PDB atom serial fields stop at 99,999 atoms. Systems above that size
  require another visualization format; segID solves residue/chain mapping but
  not the atom-serial limit.
- The current root mBuild hierarchy and Interchange representation, rather than
  HOOMD DPD itself, appear likely to be the limiting memory cost at 800k.

## Scientific diagnosis: same idea, different protocol

It would be inaccurate to conclude that DPD cannot initialize AA melts. The
adaptive protocol has produced finite, minimizable, dynamically stable AA
systems through 80k atoms at high mass density. It would also be inaccurate to
claim that Eric's CG protocol transfers unchanged.

The transferable part is the algorithmic idea:

`overlapping periodic random walks -> soft DPD relaxation -> FIRE -> production force field`

The non-transferable parts are the numerical constants and fixed iteration
count. Explicit atoms raise reduced number density from roughly 1 or less to
4.4--5.5, and local geometry is governed by heterogeneous high-frequency
bonded terms. Rings, impropers, side chains, and 1-3/1-4 topology create force
competition absent from a linear bead-spring model.

The evidence therefore supports presenting AA-DPD as a separate parameterized
regime or protocol within PhantomWalk, not as the CG protocol applied at atom
resolution. The AA result is scientifically stronger if the paper states this
limitation explicitly.

## Recommended next experiments

1. Repeat adaptive 40k PE, P3HT, and PES at three or more seeds.
2. Complete adaptive 80k P3HT and PES with minimization and 2 fs NVT/NPT.
3. Run at least one true branched-backbone chemistry.
4. On a 48--64 GiB host, run the planned approximately 800k PE case and retain
   the full timing, memory, energy, distance, NVT, and NPT record.
5. Extend selected NVT/NPT trajectories to at least tens of picoseconds before
   drawing equilibrium or density conclusions.
6. Compare local bond, angle, proper, and improper distributions immediately
   after DPD/FIRE, after minimization, and after short NVT.
7. Quantify coordinate displacement or RMSD during minimization in addition to
   endpoint energy; energy alone does not identify which structural degrees of
   freedom move.
8. Consider a size-independent extreme-contact diagnostic alongside the
   intensive energy rule, since one rare contact can matter even when averages
   are converged.

## Reproducible artifacts

- `benchmark_results/aa_iteration_sweep/`: fixed DPD/FIRE sweep, robustness
  runs, 10k finalists, and plots.
- `benchmark_results/aa_adaptive_scaling/`: adaptive scaling, 2 fs dynamics,
  raw JSON Lines, and scaling plot.
- `phantomwalk/benchmarks/README.md`: rejected bonded-scaling experiment.
- `phantomwalk/lib/fastfire.py`: current AA-DPD and FIRE parameters.
- `phantomwalk/lib/all_atom.py`: Sage/Interchange handoff, uncapped minimizer,
  and production NVT/NPT validation.

## References

- [OpenFF Interchange example using Sage 2.3.0](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-interchange/ligand_in_water/ligand_in_water.html)
- [OpenFF example using NAGL graph-network charges](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-interchange/host-guest/host_guest.html)
- [HOOMD-blue DPD force documentation](https://hoomd-blue.readthedocs.io/en/v3.0.0/module-md-pair.html)
- [HOOMD-blue FIRE convergence documentation](https://hoomd-blue.readthedocs.io/en/v5.2.0/hoomd/md/minimize/fire.html)
- [OpenMM LocalEnergyMinimizer and MinimizationReporter API](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html)
- [OpenMM Langevin and pressure-coupling guidance](https://docs.openmm.org/7.5.0/userguide/application.html)
