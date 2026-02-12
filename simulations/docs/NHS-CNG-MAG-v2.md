# NHS-CNG-MAG-v2

## Model

Commuting multi-qubit magnetization sectors:

- Total magnetization eigenvalues: `m in {-N, -N+2, ..., N}`
- Sector degeneracy: `g(m) = C(N, (N+m)/2)`
- Exact sector law:
  `p(m) propto g(m) exp(k2 m^2 + k4 m^4)`

This is the V1 multi-qubit magnetization benchmark carried into V2.

## Why this test

1. Exact closed-form reference exists at every point.
2. Includes both PSD (`k4>0`) and non-PSD (`k4<0`) branches.
3. Non-PSD branch is a direct contour-rotation stress test.

## Implementation

HS Monte Carlo estimator with auxiliary variables:

- `X ~ N(0, 2 k2)` (or complex-rotated equivalent)
- `Y ~ N(0, 8 k4)` (real for PSD, rotated contour for non-PSD)
- Unnormalized sector weights:
  `w_m = g(m) E[exp(m X + 0.5 m^2 Y)]`

Reported diagnostics:

1. `trace_distance` between analytic and MC sector distributions.
2. `max_abs_error` over sectors.
3. Imaginary leakage:
   `sum_m |Im w_m| / sum_m |Re w_m|`.
4. Convergence vs sample count `M`.

## Figure mapping

- `nhs_synth_nonpsd_v2.pdf` (manuscript path)
- `nhs_multiqubit_mag_v2.pdf` (explicit alias)

Panels:

1. PSD branch: exact vs MC sector probabilities.
2. Non-PSD branch: exact vs contour-rotated MC.
3. Log-log convergence vs `M`.
