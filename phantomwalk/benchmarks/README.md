# All-atom benchmark notes

The current all-atom AA-DPD initializer is the physical-Angstrom FastFIRE
protocol documented in the repository README and demonstrated in
`phantomwalk/examples/4-aa-dpd-fastfire.ipynb`. The reduced-unit experiments
below are historical negative results. They are retained to explain discarded
parameter choices and should not be used as current settings or compared
directly to final physical-unit timings. In the final protocol, physical
Angstrom, amu, and kcal/mol conventions are used, but `kT`, `dt`, `gamma`, and
the DPD trajectory remain empirical initialization controls rather than
calibrated physical dynamics.

## Historical discarded reduced-unit bonded scaling experiment

On 2026-08-04, we tested two alternatives to the then-current reduced-unit
bonded scaling in a 500-step DPD plus 200-step FIRE protocol:

1. `uniform`: set every bond, angle, proper-torsion term, and improper-torsion
   term to a dimensionless force constant of 250,000.
2. `class_normalized`: preserve the relative OpenFF force constants within each
   bonded class while scaling the largest value in every class to 250,000.

The tests used approximately 10,000 atoms, seed 11, and the highest-density
state point for each chemistry. Both alternatives kept all bond force constants
at 250,000 and used the old reduced units, timestep, DPD repulsion, and
friction.

| Chemistry | Density (g/cm^3) | Policy | Result |
| --- | ---: | --- | --- |
| PE | 1.1 | uniform | Unstable during DPD; a particle left the box after 21.27 s. |
| PE | 1.1 | class-normalized | Unstable during DPD; a particle left the box after 13.93 s. |
| P3HT | 1.1 | uniform | Unstable during DPD; a particle left the box after 28.02 s. |
| P3HT | 1.1 | class-normalized | DPD/FIRE completed, but the handoff began at 1.72e20 kJ/mol and OpenMM minimization took 76.47 s. |
| PES | 1.3 | uniform | DPD/FIRE completed; the run was stopped during OpenMM minimization after the earlier results rejected the policy. |
| PES | 1.3 | class-normalized | Not run. |

The class-normalized P3HT structure did minimize to a finite energy of
10,999.76 kJ/mol, but its FastFIRE bond diagnostic was poor: 1.54 A RMS error
and 51.89 A maximum error before minimization. The maximum can include a
periodic unwrapping discontinuity, but the RMS error and enormous initial Sage
energy independently show that the handoff was not improved.

Conclusion: assigning a reduced force constant of 250,000 to angular and
torsional terms was too stiff for the old timestep and short reduced-unit
protocol. The experimental code paths were removed. The final protocol instead
uses physical Angstrom coordinates and force lengths, true masses, OpenFF 2.3.0
bonded terms with all coefficients multiplied by 30 only during DPD/FIRE,
vdW-epsilon-scaled DPD pairs with `A_base = 5000`, and an unscaled OpenFF
Interchange for OpenMM.
