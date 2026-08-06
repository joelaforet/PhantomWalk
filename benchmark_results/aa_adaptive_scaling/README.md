# Adaptive all-atom initialization scaling

## Protocol

The adaptive protocol runs a 2000-step DPD baseline followed by 250-step
extensions. It stops after every intensive force-energy component changes by
no more than 2% for two consecutive checks. It then runs at least 100 FIRE
steps, continuing FIRE when required by its convergence criterion.

The handoff validation uses uncapped OpenMM minimization with OpenFF Sage
2.3.0 and AshGC charges on CUDA. Production stability tests run 1000 steps of
NVT followed by 1000 steps of NPT at 300 K, 1 bar, and a 2 fs timestep.

## Scaling results

| Chemistry | Atoms | DPD steps | DPD time (s) | Closest nonexcluded (A) | Initial OpenMM (kJ/mol/atom) | Final OpenMM (kJ/mol/atom) |
|---|---:|---:|---:|---:|---:|---:|
| PE | 10,268 | 3,500 | 63.18 | 1.89 | 9.72 | 3.23 |
| PE | 40,166 | 3,500 | 263.12 | 1.81 | 11.03 | 4.11 |
| PE | 80,030 | 3,500 | 443.07 | 1.79 | 11.53 | 4.33 |
| P3HT | 10,016 | 4,750 | 67.68 | 2.02 | 78.33 | 1.27 |
| P3HT | 40,064 | 4,500 | 274.73 | 2.03 | 36.26 | 1.21 |
| PES | 11,136 | 5,250 | 89.28 | 2.17 | 28.73 | 1.63 |
| PES | 40,368 | 5,750 | 267.14 | 2.05 | 28.80 | 1.58 |

The required DPD iteration count is chemistry-dependent but does not increase
systematically with atom count over the tested range. At fixed density, wall
time scales approximately linearly with atom count because each DPD step is
linear-time. These data do not support making DPD iterations proportional to
atom count.

## Production dynamics at 2 fs

| Chemistry | Atoms | Finite NVT/NPT | Final NPT temperature (K) | Final NPT density (g/cm3) | Sampled potential-energy range (kJ/mol/atom) |
|---|---:|---:|---:|---:|---:|
| PE | 2,114 | yes | 298.0 | 1.019 | 4.51--5.58 |
| P3HT | 2,504 | yes | 306.3 | 0.998 | 3.15--3.85 |
| PES | 2,784 | yes | 302.6 | 1.165 | 3.63--4.60 |
| PE | 80,030 | yes | 312.3 | 1.008 | 5.20--6.52 |

The short NPT runs demonstrate stability, not equilibrium density. Their smooth
density relaxation is expected because the initial densities intentionally
include high-density conditions.

## 800k feasibility

The 80,030-atom end-to-end PE case required 4.10 GiB peak host memory on this
machine. A linear fit to the measured PE memory at 2k, 10k, 40k, and 80k
projects 29.4 GiB at 800k atoms. The current workstation has 16 GiB host RAM,
so launching the 800k Interchange/OpenMM handoff here would likely invoke the
system OOM killer and was not attempted.

An 800k validation should use at least 32 GiB available host memory; 48--64 GiB
is recommended to leave margin for OpenFF Interchange construction, OpenMM
system conversion, and analysis. The measured DPD scaling predicts roughly
74 minutes for 3500 PE DPD steps on the current CPU, excluding construction,
parameterization, minimization, and dynamics.
