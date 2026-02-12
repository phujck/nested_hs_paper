# Objection Matrix (nested_hs_paper)

| Objection | Test | Quantitative criterion |
|---|---|---|
| Non-PSD kernels break HS sampling | NHS-CNG-1 | non-PSD trace distance remains controlled and leakage stays small |
| Many-body stress invalidates the method | NHS-CNG-1 | performance sustained up to `N=16` |
| Missing non-commuting evidence | NHS-NCG-1 | off-diagonal HMF norm grows with coupling |
| K4 layer does not matter in full case | NHS-NCNG-1 | median `improvement_ratio > 1` |
