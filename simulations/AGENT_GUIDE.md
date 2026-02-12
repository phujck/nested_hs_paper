# Nested HS Simulation Agent Guide

This suite compares explicit exact reference states to explicit predictions in all four regimes.
The exact reference is always stated as either:
- `rho_S^ED = Tr_B[e^{-beta H_tot}] / Tr[e^{-beta H_tot}]`, or
- an exact finite weighted mixture over a traced static field distribution.

## Regime Models

1. NHS-CG-1 (commuting Gaussian)
- qutrit system: `H_S=diag(0,0.7,1.6)`, `f=diag(0,1,2)`.
- Harmonic bath with linear coupling.
- Prediction: commuting closed form `rho_pred propto exp[-beta(H_S-lambda f^2)]`.

2. NHS-CNG-1 (commuting non-Gaussian, many-body)
- Magnetization-sector model with exact analytic probabilities
  `p(m) propto g(m) exp(k2 m^2 + k4 m^4)`.
- HS Monte Carlo estimator is compared directly to this exact analytic distribution.
- Includes PSD (`k4>0`) and non-PSD (`k4<0`, contour-rotated) branches.

3. NHS-NCG-1 (non-commuting Gaussian)
- Non-commuting qutrit with bosonic linear coupling.
- Exact reference: ED trace-out.
- Prediction: Gaussian-field quadrature reconstruction with matched second-cumulant scale.

4. NHS-NCNG-1 (non-commuting non-Gaussian)
- Non-commuting qutrit with exact static spin-induced field mixture.
- Predictions: K2-only and K2+K4 reconstructions.

## Commands

```powershell
py -3 simulations/src/run_nested_hs_suite.py --regime all --profile full --seed 42
py -3 simulations/src/plot_nested_hs_suite.py
py -3 simulations/src/validate_nested_hs_claims.py
```

## Output Contract

- Data: `simulations/results/data/*.csv`
- Figures: `simulations/results/figures/nhs_*.pdf` and `.png`
- Metrics: `simulations/results/claim_metrics_nested_hs.json`
- Manifest: `simulations/results/manifest.json`

## Figure Semantics

- Main panels are always ED (or exact analytic) vs predicted overlays.
- Companion panels are residuals (`trace_distance`), stability (`imag_leakage`), or improvement ratios.

## Failure Modes

- CG residual too large: raise boson cutoff.
- CNG non-PSD instability: increase `M` and inspect leakage trend.
- NCG mismatch: increase quadrature order and verify variance matching.
- NCNG weak improvement: reduce coupling to remain in low-order cumulant regime.
