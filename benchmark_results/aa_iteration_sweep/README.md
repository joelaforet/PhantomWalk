# All-atom DPD/FIRE iteration sweep

## Recommendation

Use **1000 DPD steps followed by 100 FIRE steps** for the dense all-atom
initialization protocol. This was the fastest tested setting that retained a
useful safety margin at approximately 10,000 atoms for PE, P3HT, and PES and
then minimized successfully with the uncapped OpenMM minimizer on CUDA.

The 100th FIRE step is inexpensive relative to the DPD phase. At 10k atoms,
1000/100 and 1000/50 had effectively identical mean FastFIRE wall times
(25.71 and 25.72 s). For P3HT, 100 FIRE steps improved the minimum 1--4
distance from 0.45 to 0.54 Angstrom and reduced initial OpenMM energy from
7.97e6 to 8.86e5 kJ/mol/atom.

## Screen and validation criteria

The screening grid used PE and P3HT at 1.1 g/cm3 and PES at 1.3 g/cm3.
A result was classified as sensible when it had finite coordinates, minimum
nonexcluded spacing >= 0.75 Angstrom, minimum 1--4 endpoint distance >= 0.35
Angstrom, initial OpenMM energy <= 1e7 kJ/mol/atom, final minimized energy <=
10 kJ/mol/atom, and an energy decrease under uncapped CUDA/OpenMM
minimization.

The broad grid contained 48 runs: DPD steps {250, 500, 1000, 2000} crossed
with FIRE steps {50, 100, 200, 400} for the three chemistries at approximately
2000 atoms. Pareto candidates were repeated with seeds 11, 22, and 33 before
the finalists were scaled to approximately 10,000 atoms.

## Approximately 10k-atom finalists

| DPD/FIRE | Passed | Mean FastFIRE (s) | Worst initial energy (kJ/mol/atom) | Min nonexcluded (A) | Min 1--4 (A) |
|---|---:|---:|---:|---:|---:|
| 250/400 | 1/3 | 17.23 | 3.25e7 | 0.587 | 0.243 |
| 1000/50 | 3/3 | 25.72 | 7.97e6 | 0.956 | 0.446 |
| **1000/100** | **3/3** | **25.71** | **8.86e5** | **1.146** | **0.536** |

The tempting 250/400 setting passed all three chemistries at 2k atoms and all
three tested seeds, but failed for both P3HT and PES at 10k atoms. FIRE cannot
reliably replace the missing DPD relaxation: P3HT retained a 0.59 Angstrom
nonexcluded contact and PES developed a collapsed 0.24 Angstrom 1--4 distance.

## Interpretation

DPD steps control the size-robust removal of the random-walk overlaps. FIRE is
cheap cleanup after that process, but adding FIRE iterations does not rescue a
DPD phase that is too short. Beyond 1000 DPD steps, the small-system grid found
diminishing structural benefit relative to wall time. FIRE frequently reached
the current numerical convergence criterion by 100 steps in the small systems,
although P3HT and PES at 10k did not; the recommendation is therefore a fixed,
fast initialization protocol validated by geometry and OpenMM handoff, not a
claim that FIRE is fully converged for every system.

The reported FastFIRE timing excludes polymer construction, OpenFF/AshGC
parameterization, diagnostics, and OpenMM minimization. Those stages are
recorded separately in the JSONL output.
