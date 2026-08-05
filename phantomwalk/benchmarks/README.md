# All-atom benchmark notes

## Bonded scaling experiment

On 2026-08-04, we tested two alternatives to the working MuPT-style bonded
scaling in the 500-step DPD plus 200-step FIRE protocol:

1. `uniform`: set every bond, angle, proper-torsion term, and improper-torsion
   term to a dimensionless force constant of 250,000.
2. `class_normalized`: preserve the relative OpenFF force constants within each
   bonded class while scaling the largest value in every class to 250,000.

The tests used approximately 10,000 atoms, seed 11, and the highest-density
state point for each chemistry. Both alternatives kept all bond force constants
at 250,000 and used the existing reduced units, timestep, DPD repulsion, and
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
torsional terms is too stiff for the current timestep and short protocol. The
experimental code paths were removed. Continue using the working implementation:
uniform dimensionless bond constants of 250,000, with OpenFF angle, proper, and
improper strengths divided by the reference LJ epsilon and multiplied by the
MuPT bonded scale of 30.
