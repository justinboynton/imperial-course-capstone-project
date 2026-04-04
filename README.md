# Capstone Project — Bayesian Optimisation of Black-Box Functions

## Overview

This repository documents my participation in the Stage 2 capstone challenge: maximising eight unknown black-box functions using Bayesian optimisation with a budget of one query per function per week.

Each function accepts a normalised input vector (all dimensions in [0, 1]) and returns a single scalar output. The internal mechanics of each function are unknown — only the inputs and returned outputs are observable.

---

## Approach

The core strategy uses a **Gaussian Process (GP) surrogate model** fitted on all available observations (provided initial data + weekly portal submissions), combined with an **acquisition function** to select the next query point.

The acquisition function balances:
- **Exploration** — querying uncertain regions the GP has not seen
- **Exploitation** — querying regions the GP predicts are high-valued

Settings (acquisition function, β, ξ, kernel) are tuned per function based on the landscape characteristics revealed by the data.

A Streamlit dashboard (`capstone_app.py`) provides:
- GP-guided query suggestions
- GP posterior visualisation (mean ± 95% CI slices)
- Acquisition function comparison plots
- AI-powered analysis via Claude

---

## Function Summary

| Fn | Dims | Description | Initial Best Y | Week 1 Y | Current Best Y | Acq | Kernel |
|----|------|-------------|---------------|----------|---------------|-----|--------|
| 1  | 2D   | Contamination/radiation field | ≈0.000 | ≈0.000 | ≈0.000 | UCB β=2.0 | Matérn 5/2 |
| 2  | 2D   | Noisy log-likelihood | 0.6112 | 0.0259 | 0.6112 | UCB β=2.5 | Matérn 5/2 |
| 3  | 3D   | Drug compound combinations | -0.0348 | -0.0417 | -0.0348 | EI ξ=0.02 | Matérn 5/2 |
| 4  | 4D   | Warehouse ML hyperparameters | -4.0255 | -21.254 | -4.0255 | UCB β=2.0 | Matérn 5/2 |
| 5  | 4D   | Chemical yield (unimodal) | 1088.86 | 50.44 | 1088.86 | EI ξ=0.01 | RBF |
| 6  | 5D   | Cake recipe (negative penalty) | -0.7143 | -1.8257 | -0.7143 | EI ξ=0.02 | Matérn 5/2 |
| 7  | 6D   | GBM hyperparameter tuning | 1.3650 | 0.1207 | 1.3650 | EI ξ=0.05 | Matérn 5/2 |
| 8  | 8D   | Complex ML hyperparameters | 9.5985 | 9.2597 | 9.5985 | UCB β=2.5 | Matérn 5/2 |

*Current Best Y = best of initial data and all portal submissions combined.*

---

## Repository Structure

```
├── capstone_app.py              # Streamlit optimisation dashboard
├── capstone_history.json        # Observation store (updated each week)
├── capstone_optimiser.py        # Standalone optimiser script
├── initial_data/                # Provided initial .npy observations
│   ├── function_1/ ... function_8/
├── weekly_snapshots/            # Per-week submission records
│   └── week_01.json
├── README.md                    # This file
├── MODEL_CARD.md                # Per-function surrogate model decisions
├── REFLECTIONS.md               # Weekly reflections (all functions)
└── STRATEGY.md                  # High-level strategy evolution
```

---

## Running the App

```bash
pip install streamlit numpy pandas scikit-learn scipy plotly anthropic
streamlit run capstone_app.py
```

Set your Anthropic API key in `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Weekly Progress Log

| Week | Date | Key Action | Key Learning |
|------|------|------------|--------------|
| 1 | 2026-04-04 | Initial submissions across all 8 functions | Initial .npy data not yet integrated into GP — all queries below initial best. Fixed for week 2. |

---

## Branch History

| Branch | Description |
|--------|-------------|
| `main` | Current working state |
| `week-1` | Snapshot after week 1 submissions |


# Contact
Justin Boynton <justin.boynton@gmail.com>
