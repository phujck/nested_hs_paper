# NHS-NCG-1

## Hamiltonian
- Non-commuting qutrit: `H_S=diag(0,0.9,1.8)`,
  `f=lambda_1+0.45 lambda_6+0.2 lambda_4`.
- Bosonic bath with linear coupling.

## Exact reference
`rho_S^ED = Tr_B[e^{-beta H_tot}] / Tr[e^{-beta H_tot}]` from ED.

## Prediction
Gaussian-field quadrature reconstruction
`rho_S^pred = E_x[e^{-beta(H_S + x f)}]/Tr[...]` with matched variance.

## What is plotted
- ED and prediction overlay for `p0(coupling)`.
- Residual trace distance vs coupling.
