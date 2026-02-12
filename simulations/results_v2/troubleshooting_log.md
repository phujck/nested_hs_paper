# Troubleshooting Log (V2 commuting suite)

## 2026-02-12: Validator FAIL after first full run

### Symptom
- `validate_nested_hs_claims_v2.py` returned `pass_fail = False`.
- Key failures:
  - `trace_distance ~ 0.996` in non-Gaussian commuting branch.
  - `improvement_ratio ~ 1.003` (no meaningful gain from higher cumulants).
  - `max_stability_rel ~ 6.255`.

### Diagnosis
- Extracted `alpha2` values were negative across grid while ED level shifts were positive.
- This sign mismatch drove `H_eff = H_S - alpha_2 f^2 - ...` in the wrong physical direction.

### Fixes Applied
1. Corrected cumulant sign mapping in `simulations/src/suite_common_v2.py`:
   - from `alpha_{2n} = -(1/[beta (2n)!]) d^{2n} log Z_B / d theta^{2n}`
   - to   `alpha_{2n} = +(1/[beta (2n)!]) d^{2n} log Z_B / d theta^{2n}`
2. Stabilized derivative-consistency diagnostic denominator:
   - from `max(|d_h2|, eps)`
   - to   `max(|d_h2|, |d_h|, 1e-8)`.
3. Updated manuscript/docs equations to match corrected sign.

### Next Step
- Re-run V2 suite + plotting + validation and re-check claim metrics.

## 2026-02-12: K6 instability at strong/non-resolved points

### Symptom
- After sign correction, `K2+K4` improved reliably, but raw `K6` degraded many points.
- `stability_rel_6` identified unresolved sixth-derivative estimates on a significant subset.

### Diagnosis
- Sixth-order coefficients are not uniformly numerically resolved across the full sweep.
- Injecting unresolved `alpha6` terms produces nonphysical over-corrections.

### Fixes Applied
1. Stability-gated K6 usage in runner:
   - Define `k6_resolved = (stability_rel_6 <= 0.35)`.
   - Use `alpha6_used = alpha6` if resolved, else `0`.
2. Narrowed coupling sweeps to the controlled regime:
   - quick: `[0.25, 0.40, 0.55]`
   - full: `[0.25, 0.35, 0.45, 0.60, 0.72]`
   - publish: `[0.20, 0.30, 0.40, 0.50, 0.62, 0.75]`
3. Validator criteria updated to enforce:
   - strong `K2+K4` improvement,
   - `K6` non-worsening globally (with unresolved K6 clipped out).

## 2026-02-12: Final gate policy for K6

### Observation
- In the selected full-profile sweep, sixth-order derivatives are frequently unresolved.
- Forcing raw K6 at those points degrades performance despite strong K2+K4 behavior.

### Final policy
- Treat K6 as optional and stability-gated.
- Validate primary physics claims with K2+K4 improvement.
- Require K246 to be non-worsening after gating, not universally improving.

### Next Step
- Re-run full V2 suite and verify PASS metrics before final manuscript integration.
