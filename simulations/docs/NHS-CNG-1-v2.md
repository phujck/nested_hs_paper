# NHS-CNG-1-v2

## Model

Commuting qutrit:
- `H_S = diag(0, 0.9, 1.8)`
- `f = diag(0, 1, 2)`

Finite-spin bath:
- `H_B = sum_j omega_j sigma_j^x`
- `B = sum_j c_j sigma_j^z`
- `H_I = g f ⊗ B`

This keeps linear coupling in bath operator `B`; non-Gaussianity comes from bath statistics.

## Exact Reference and Truncation Predictions

- Exact reference:
  `rho_S^ED = Tr_B[e^{-beta(H_S+H_B+g f⊗B)}]/Tr[e^{-beta(...) }]`.
- Source-deformed partition:
  `Z_B(theta)=Tr_B[e^{-beta(H_B+theta B)}]`.
- Integrated cumulant coefficients:
  `alpha_{2n}=(1/[beta (2n)!]) d^{2n} log Z_B(theta)/d theta^{2n} |_{theta=0}`.
- Truncation ladder:
  - `K2`: `H_eff = H_S - alpha2 f^2`
  - `K2+K4`: `H_eff = H_S - alpha2 f^2 - alpha4 f^4`
  - `K2+K4+K6`: add `-alpha6 f^6`.
- Stability gate:
  - use `alpha6_used = alpha6` only if `stability_rel_6 <= 0.35`,
  - otherwise set `alpha6_used = 0` so `K246` is non-worsening.

## Plotted Quantities

- Level populations `p0,p1,p2` (ED vs truncations).
- Trace distances `D_K2`, `D_K24`, `D_K246` to ED.
- Improvement ratios:
  - `D_K2 / D_K24`
  - `D_K2 / D_K246`
- Cumulant-ratio controls:
  - `chi4 = alpha4/alpha2^2`
  - `chi6 = alpha6/|alpha2|^3`.

## Objections Answered

- “Quartic term is ad hoc”: coefficients come from explicit bath thermodynamics via `Z_B(theta)`.
- “Higher orders are noise”: ladder comparison checks systematic error reduction.
- “Finite bath is too small to matter”: exact trace-out still gives nontrivial cumulant hierarchy and measurable corrections.

## Acceptance Criteria

- Majority of points satisfy `D_K24 < D_K2`.
- Majority of points satisfy `D_K246 < D_K24`.
- Median `D_K2/D_K246 > 1`.
- Derivative-stability diagnostics (`stability_rel_2,4,6`) remain bounded.
