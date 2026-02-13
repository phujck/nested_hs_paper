---
description: Workflow for running simulations and generating figures.
---

# Simulation Pipeline

## 1. Source Code (`simulations/src/`)
- Write modular Python scripts.
- Use `argparse` for parameter sweeps (e.g., `--beta`, `--g`).
- **Output**: Save raw data (CSV/NPZ) to `simulations/results/data/`. Do not commit large data files.

## 2. Plotting
- Separate computation from visualization.
- Create specific plotting scripts (e.g., `plot_fig1.py`).
- Read data from `simulations/results/data/`.
- **Output**: Save figures to `simulations/results/figures/` as PDF.

## 3. Automation (`manage.ps1`)
- Use a script runner to orchestrate.
- Example command: `./manage.ps1 plots` should regenerate all figures from data.

## 4. Reproducibility
- Record random seeds if stochastic.
- Store parameter metadata in output filenames or headers.
