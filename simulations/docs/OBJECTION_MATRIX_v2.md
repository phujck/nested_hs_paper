# Objection Matrix (nested_hs\_paper V2)

| Objection | Test | Quantitative criterion |
|---|---|---|
| V1 multi-qubit benchmark was dropped in V2 | `NHS-CNG-MAG-v2` | explicit V2 figure present (`nhs_multiqubit_mag_v2.pdf`) with PSD and non-PSD overlays |
| Contour-rotated non-PSD sampling is not reliable | `NHS-CNG-MAG-v2` | non-PSD branch trace distance remains below threshold and decreases with `M` |
| Gaussian commuting agreement is cosmetic | `NHS-CG-1-v2` | `td_ed_analytic` decreases with cutoff and remains below threshold at max cutoff |
| HS sampling does not converge | `NHS-CG-1-v2` | log-log slope of MC error vs `M` is within a `1/sqrt(M)` band |
| Qutrit phase handling is not demonstrated | `NHS-CNG-PHASE-v2` | phase scan has bounded trace-distance error and finite replica uncertainty bands |
| Non-Gaussian coefficients are fitting parameters | `NHS-CNG-1-v2` | `alpha2, alpha4, alpha6` computed from `log Z_B(theta)` derivatives only |
| Quartic correction is arbitrary | `NHS-CNG-1-v2` | `D_K24 < D_K2` on most grid points |
| Higher-order corrections are unstable | `NHS-CNG-1-v2` | K6 is stability-gated (`stability_rel_6 <= 0.35`), and gated `K246` is non-worsening vs `K24` |
| Method hides non-commuting limitations | manuscript scope | explicit statement deferring non-commuting sector to final paper |
