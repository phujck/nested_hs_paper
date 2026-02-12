# NHS-CNG-PHASE-v2

## Model

Commuting qutrit clock with complex quartic phase:

- `kappa4 = r exp(i phi)`
- `kappa2 = conj(kappa4)`
- Phase scanned as `phi/pi in [-1,1]`
- Correlation-length labels: `tau_c in {0.04, 0.12, 0.35}`

## Why this test

1. Directly stress-tests complex-phase handling in HS sampling.
2. Produces nontrivial periodic structure in qutrit populations.
3. Connects to the V1 clock-style phase plots.

## Implementation

For each `(phi, tau_c)`:

1. Compute analytic populations from the commuting expression.
2. Run replica-averaged functional HS Monte Carlo with fixed total sample budget.
3. Record mean and standard deviation for each population component.
4. Report mean and standard deviation of trace distance and max component error.

## Figure mapping

- `nhs_cng_phase_clock_v2.pdf`

Panels:

1. Smooth analytic phase curves with MC overlays (representative `tau_c`).
2. Trace-distance error vs phase for each `tau_c`, with uncertainty bands.
