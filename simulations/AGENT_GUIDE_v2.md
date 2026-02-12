# Nested HS V2 Agent Guide (Commuting-Only)

This V2 track keeps the commuting focus and restores the V1 commuting benchmarks in a clearer order.

## V1 to V2 Parity

1. `NHS-CG-1` (V1 Gaussian qutrit) -> retained as `NHS-CG-1-v2`.
2. `NHS-CNG-1` (V1 multi-qubit magnetization sectors) -> retained in V2 as `NHS-CNG-MAG-v2`.
3. V1 qutrit phase stress test -> retained in V2 as `NHS-CNG-PHASE-v2`.
4. New in V2: finite-spin bath ED truncation ladder -> `NHS-CNG-1-v2`.
5. V1 non-commuting tests (`NHS-NCG-1`, `NHS-NCNG-1`) -> intentionally deferred to the final paper.

## Tests in V2

1. `NHS-CNG-MAG-v2` (multi-qubit magnetization sectors)
- Exact sector law: `p(m) propto g(m) exp(k2 m^2 + k4 m^4)`.
- Includes PSD and non-PSD (`k4 < 0`) contour-rotation branch.
- Purpose: algorithmic stress test and direct continuity with V1.

2. `NHS-CNG-PHASE-v2` (qutrit clock phase benchmark)
- Complex phase scan in quartic cumulant with replica-averaged MC points.
- Purpose: verify stable phase handling in commuting non-Gaussian sampling.

3. `NHS-CG-1-v2` (commuting Gaussian qutrit, finite-mode bosonic ED)
- ED vs analytic `rho propto exp[-beta(H_S - lambda f^2)]` plus HS estimators.
- Purpose: exact commuting Gaussian baseline and MC scaling.

4. `NHS-CNG-1-v2` (commuting non-Gaussian qutrit, finite-spin bath ED)
- Truncation ladder: `K2`, `K2+K4`, stability-gated `K2+K4+K6`.
- Purpose: physically computed cumulant hierarchy vs exact reduced state.

## Commands

```powershell
py -3 simulations/src/run_nested_hs_suite_v2.py --regime all --profile publish --seed 42
py -3 simulations/src/plot_nested_hs_suite_v2.py
py -3 simulations/src/validate_nested_hs_claims_v2.py
```

## Output Map

- Data:
  - `simulations/results_v2/data/cg_nhs_cg_1_v2.csv`
  - `simulations/results_v2/data/cng_nhs_cng_1_v2.csv`
  - `simulations/results_v2/data/cng_nhs_synth_nonpsd_v2.csv`
  - `simulations/results_v2/data/cng_nhs_synth_nonpsd_curves_v2.csv`
  - `simulations/results_v2/data/cng_nhs_phase_clock_v2.csv`
- Figures:
  - `simulations/results_v2/figures/nhs_synth_nonpsd_v2.pdf`
  - `simulations/results_v2/figures/nhs_multiqubit_mag_v2.pdf`
  - `simulations/results_v2/figures/nhs_cng_phase_clock_v2.pdf`
  - `simulations/results_v2/figures/nhs_cg_1_v2.pdf`
  - `simulations/results_v2/figures/nhs_cng_1_v2.pdf`
- Metrics:
  - `simulations/results_v2/claim_metrics_nested_hs_v2.json`
- Manifest:
  - `simulations/results_v2/manifest.json`

## Common Failure Modes

1. MC slope check fails in publish profile.
- Cause: noisy single-seed scaling points.
- Fix: use built-in replica averaging for scaling points (already enabled in V2).

2. `K6` does not improve.
- Cause: derivative instability in finite-difference extraction.
- Interpretation: expected in this finite-spin setup; `K6` is stability-gated.

3. User cannot find V1 multi-qubit content.
- Use figure alias `nhs_multiqubit_mag_v2.pdf`.
- See manuscript subsection "Multi-qubit magnetisation benchmark (NHS-CNG-MAG-v2)".
