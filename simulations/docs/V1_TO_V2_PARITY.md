# V1 to V2 Parity Map

This document tracks how each V1 numerical element is carried into the commuting-only V2 track.

## Benchmarks

1. `NHS-CG-1` (V1)
- Status in V2: retained as `NHS-CG-1-v2`.
- Data: `cg_nhs_cg_1_v2.csv`.
- Figure: `nhs_cg_1_v2.pdf`.

2. `NHS-CNG-1` (V1 multi-qubit magnetization sectors)
- Status in V2: retained as `NHS-CNG-MAG-v2`.
- Data: `cng_nhs_synth_nonpsd_v2.csv`, `cng_nhs_synth_nonpsd_curves_v2.csv`.
- Figures: `nhs_synth_nonpsd_v2.pdf`, alias `nhs_multiqubit_mag_v2.pdf`.

3. V1 qutrit clock phase stress test
- Status in V2: retained as `NHS-CNG-PHASE-v2`.
- Data: `cng_nhs_phase_clock_v2.csv`.
- Figure: `nhs_cng_phase_clock_v2.pdf`.

4. V1 non-commuting tests (`NHS-NCG-1`, `NHS-NCNG-1`)
- Status in V2: deferred by design to the final paper (`what_rules_equilibrium` track).

## New V2 addition

1. `NHS-CNG-1-v2` finite-spin bath ED benchmark
- Purpose: physically computed non-Gaussian commuting truncation ladder with exact reduced-state reference.
- Data: `cng_nhs_cng_1_v2.csv`.
- Figure: `nhs_cng_1_v2.pdf`.
