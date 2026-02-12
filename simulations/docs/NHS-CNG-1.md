# NHS-CNG-1

## Model
Magnetization-sector probabilities
`p(m) propto g(m) exp(k2 m^2 + k4 m^4)` for `N` qubits.

## Exact reference
Closed-form analytic sector probabilities from the expression above.

## Prediction
Nested HS Monte Carlo estimator with
`X ~ N(0,2k2)`, `Y ~ N(0,8k4)`;
for `k4<0`, contour-rotated sampling is used.

## What is plotted
- PSD overlay: analytic vs HS MC sector probabilities.
- non-PSD overlay: analytic vs HS MC sector probabilities.
- Residual scaling: trace distance vs `M` at largest `N`.
