# Capstone Project — Bayesian Optimisation of Black-Box Functions

---

## Section 1: Project Overview

### What is this project?

This repository documents participation in the Stage 2 capstone challenge: **maximising eight unknown black-box functions** using intelligent search strategies and a budget of one query per function per week.

Each function represents a real-world optimisation scenario — from detecting radiation sources in a 2D field, to tuning hyperparameters of ML models, to optimising chemical yield in a factory process. The functions are called "black-box" because only the inputs and outputs are observable; the internal equations, gradients and structure are entirely unknown.

The goal is to find the input vector that produces the **highest possible output** for each function, using the fewest possible queries. This mirrors how optimisation is practised in research and industry where each evaluation is expensive — whether that evaluation is a drug trial, a simulation run, or training a large ML model.

### Why is this relevant in real-world ML?

Black-box optimisation (BBO) sits at the heart of many practical ML challenges:

- **Hyperparameter tuning** — searching for the best learning rate, regularisation strength and architecture without access to closed-form gradients
- **Neural Architecture Search (NAS)** — evaluating model designs where each trial requires full training
- **Drug discovery and materials science** — running physical experiments where each data point has a real cost
- **Scientific simulation** — e.g., climate models, protein folding, where each forward pass is computationally expensive

The core insight is that *how* you decide where to query next is itself an ML problem. Using a surrogate model (a Gaussian Process in this case) to approximate the unknown function and an acquisition function to decide where to query next is the foundation of **Bayesian Optimisation** — one of the most sample-efficient strategies known.

---

## Section 2: Inputs and Outputs

### Input format

Each function accepts a **normalised input vector** where every dimension is bounded to the interval **[0, 1]**. Inputs are submitted as a comma-separated list of floating-point values via the capstone portal.

| Function | Dimensions | Submission format example |
|----------|-----------|--------------------------|
| F1 | 2D | `0.731, 0.733` |
| F2 | 2D | `0.703, 0.927` |
| F3 | 3D | `0.493, 0.612, 0.340` |
| F4 | 4D | `0.578, 0.429, 0.426, 0.249` |
| F5 | 4D | `0.224, 0.846, 0.879, 0.879` |
| F6 | 5D | `0.728, 0.155, 0.733, 0.694, 0.056` |
| F7 | 6D | `0.058, 0.492, 0.247, 0.218, 0.420, 0.731` |
| F8 | 8D | `0.056, 0.066, 0.023, 0.039, 0.404, 0.801, 0.488, 0.893` |

**Constraints:**
- All input values must be in [0, 1]
- Each dimension is continuous — fractional values are valid and expected
- One submission per function per week
- Inputs are treated as independent; no ordering constraints apply between dimensions

### Output format

Each function returns a **single scalar value** — a real number representing the performance signal for that input. The portal returns this value after processing.

| Function | Output range (observed) | Sign | Domain context |
|----------|------------------------|------|----------------|
| F1 | ≈ 0 | Near-zero | Contamination signal — extremely localised; hotspot not yet found |
| F2 | [−0.066, 0.726] | Mixed | Noisy log-likelihood |
| F3 | [−0.399, −0.009] | Negative | Negative side-effect count; maximise toward 0 |
| F4 | [−32.6, +0.136] | Mixed | Difference from baseline; first positive result in W6 |
| F5 | [50.4, 1412.6] | Positive | Chemical yield — unimodal, one clear peak |
| F6 | [−2.57, −0.296] | Negative | Negative penalty score; maximise toward 0 |
| F7 | [0.003, 2.358] | Positive | ML model performance score |
| F8 | [5.59, 9.800] | Positive | Validation accuracy (scaled) |

---

## Section 3: Challenge Objectives

### Goal

The objective for every function is **maximisation** — find the input vector that returns the highest possible output value.

### Constraints and limitations

| Constraint | Detail |
|-----------|--------|
| **Query budget** | One submission per function per week; evaluations are irreversible |
| **Response delay** | Results are returned later in the week; no same-week iteration |
| **Unknown structure** | No gradients, no closed-form expression, no visualisation of the function surface |
| **Initial data only** | Each function starts with 10–40 provided data points; these are the only prior observations |
| **Dimensionality growth** | Functions range from 2D (F1, F2) up to 8D (F8), making exhaustive search infeasible for higher-dimensional functions |
| **Noisy outputs** | Some functions (e.g. F2) produce noisy outputs, making it harder to distinguish signal from variance |
| **Local optima risk** | Some functions have multiple peaks; finding a local maximum may not be the global one |

### What "success" looks like

A successful submission is not necessarily the global maximum — it is **a higher output than all previous submissions for that function**. Progress is measured week-on-week, and the capstone rewards thoughtful iteration and documented reasoning as much as raw performance.

---

## Section 4: Technical Approach

### Core framework: Gaussian Process Bayesian Optimisation

The primary strategy is **Bayesian Optimisation (BO)** using a **Gaussian Process (GP) surrogate model**. The GP is fitted on all available observations (initial data + all weekly portal submissions) and used to:

1. **Predict the expected output** at any unsampled point (posterior mean)
2. **Quantify uncertainty** at any unsampled point (posterior standard deviation)
3. **Guide the next query** via an acquisition function that balances exploration and exploitation

This is implemented in `capstone_app.py` (Streamlit dashboard) and supports per-function configuration of kernel, acquisition function and hyperparameters.

### Active surrogate improvements

The following enhancements have been applied on top of the baseline GP. See `MODEL_CARD.md` for the full engineering changelog.

| Feature | Functions | What it does |
|---------|-----------|-------------|
| **ARD kernel** | F4, F7, F8 | Learns a separate length-scale per input dimension; automatically down-weights irrelevant dimensions |
| **Output standardisation** | F2–F8 | Z-scores Y before fitting so β and ξ acquisition parameters have consistent meaning across all functions |
| **Heteroscedastic GP** | F2 | Assigns higher noise to the noisy peak region via LOO-estimated per-point alpha; prevents acquisition function from chasing noise-driven gradients |
| **arcsinh Y-transform** | F1 | Spreads near-zero values so the GP can detect otherwise invisible signal gradients |

### Acquisition functions used

| Function | Strategy | Rationale |
|----------|----------|-----------|
| **UCB** | `μ(x) + β·σ(x)` | Explicit explore/exploit via β; used for uncertain landscapes (F1, F2, F8) |
| **EI** | Probability-weighted gain over current best | Concentrates queries near the current best; used for structured landscapes (F3, F5, F6, F7) |
| **GP Posterior Mean** | `μ(x)` only | Pure exploitation; used for F5 once the unimodal peak was located |

### Kernel selection

| Kernel | Functions | Why |
|--------|----------|-----|
| **Matérn 5/2** | F1–F4, F6–F8 | Moderate roughness; robust for real-world functions that are smooth but not infinitely differentiable |
| **RBF** | F5 | Unimodal landscape; infinite smoothness assumption is appropriate for a single clean peak |

### Why not neural networks or SVMs?

SVMs via SVR can model smooth functions but provide no uncertainty estimate — making them unsuitable for principled acquisition functions. Neural networks were empirically evaluated (`analysis/05_nn_surrogate_analysis.ipynb`): at n ≤ 44, a Deep Ensemble (K=10) achieved LOO R²=−0.417 on F7 and 0.906 on F8, versus the GP's 0.563 and 0.985 respectively. The GP's parameter efficiency (9 hyperparameters vs 289 MLP parameters for F8) and analytically calibrated uncertainty make it the correct choice at this sample size.

### AI-assisted strategy analysis

The dashboard integrates **Anthropic Claude** to generate per-function strategy analysis. Claude receives function metadata, initial data statistics, all portal observations, and the current best result, and returns a structured recommendation. Analyses are stored persistently in `capstone_history.json`.

---

## Function Summary

| Fn | Dims | Description | Initial Best Y | Portal Best Y | Overall Best Y | Best Week |
|----|------|-------------|---------------|--------------|----------------|-----------|
| 1  | 2D   | Contamination/radiation field | ≈0.000 | ≈0.000 | ≈0.000 | — |
| 2  | 2D   | Noisy log-likelihood | 0.6112 | **0.7260** | 0.7260 | W6 ↑ |
| 3  | 3D   | Drug compound combinations | −0.0348 | **−0.0090** | −0.0090 | W5 ↑ |
| 4  | 4D   | Warehouse ML hyperparameters | −4.0255 | **+0.1362** | +0.1362 | W6 ↑ |
| 5  | 4D   | Chemical yield (unimodal) | 1088.86 | **1412.63** | 1412.63 | W5 ↑ |
| 6  | 5D   | Cake recipe (negative penalty) | −0.7143 | **−0.2956** | −0.2956 | W6 ↑ |
| 7  | 6D   | GBM hyperparameter tuning | 1.3650 | **2.3576** | 2.3576 | W2 |
| 8  | 8D   | Complex ML hyperparameters | 9.5985 | **9.8001** | 9.8001 | W5 ↑ |

**7 of 8 functions have beaten the initial data best. F4 crossed zero for the first time in W6. F1 remains the only unsolved function.**

---

## Repository Structure

```
├── capstone_app.py              # Streamlit optimisation dashboard (primary tool)
├── capstone_optimiser.py        # Legacy CLI toolkit (superseded by dashboard)
├── capstone_history.json        # Observation store — all submissions and AI analyses
├── requirements.txt             # Pinned Python dependencies
├── initial_data/                # Provided initial .npy observations (read-only)
│   └── function_1/ … function_8/
├── analysis/                    # Exploratory analysis notebooks
│   ├── 01_initial_data_eda.ipynb          # Week 1 landscape exploration
│   ├── 02_function2_svr_exploration.ipynb # F2: SVR vs GP comparison
│   ├── 03_function8_rf_surrogate.ipynb    # F8: Random Forest surrogate evaluation
│   ├── 04_function8_gpgbm_ensemble.ipynb  # F8: GP vs GBM vs ensemble comparison
│   ├── 05_nn_surrogate_analysis.ipynb     # NN/CNN viability analysis (all functions)
│   └── figures/                           # Charts generated by the notebooks above
├── weekly_snapshots/            # Per-week submission records
│   ├── week_01.json … week_05.json
├── README.md                    # This file
├── MODEL_CARD.md                # Per-function surrogate config, history and changelog
├── REFLECTIONS.md               # Weekly reflections and engineering decisions
└── STRATEGY.md                  # High-level strategy evolution log
```

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run capstone_app.py
```

Set your Anthropic API key in `.streamlit/secrets.toml`:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Weekly Progress Log

| Week | Date | Key submissions | Notable outcomes |
|------|------|-----------------|-----------------|
| 1 | 2026-04-04 | All 8 functions | All below initial best — initial .npy data not yet integrated into GP. Fixed for W2. |
| 2 | 2026-04-11 | All 8 functions | F4 (−1.18), F5 (1138.9), F7 (2.36), F8 (9.70) all beat initial best. F2, F3, F6 below initial. Over-exploration dominant error. |
| 3 | 2026-04-18 | All 8 functions | F5 new best (1374.5). F6 new best (−0.384). F2 near initial (0.493). F8 regression (7.32) — exploration query with high D2/D4. |
| 4 | 2026-04-25 | All 8 functions | F2 **beats** initial best (0.649). F8 recovered (8.28). F6 regression (−1.29) — over-exploitation in wrong region. |
| 5 | 2026-04-10 | All 8 functions | **Best week to date: 4 new all-time bests** — F3 (−0.0090), F5 (1412.6), F6 (−0.341), F8 (9.800). F7 reproduced W2 result exactly, confirming it is deterministic. F2 regression (0.513) reveals X₂ ≈ 0.96 sensitivity. F4 partial recovery but W2 best still unchallenged. F1: 5 consecutive zeros — lower-left also fails. |
| 6 | 2026-04-17 | All 8 functions | **3 new all-time bests** — F2 (0.726), F4 (+0.136, first positive ever), F6 (−0.296). F3 stable near-best (−0.013). F7 D1 perturbation test (2.189) confirms D1=0.095 is optimal. F5 regression (1223) — D4 violated. F8 regression (9.189) — UCB pushed D1/D4 outside hard constraints. F1: sixth consecutive zero; cluster abandoned. |

---

## Recent Engineering Changes

The following improvements have been made to the surrogate pipeline since Week 1. Full details are in `MODEL_CARD.md` under *Engineering Changes Log*.

| Change | Applied | Scope |
|--------|---------|-------|
| ARD kernels | Between W1–W2 | F4, F7, F8 |
| Output standardisation (`standardize` Y-transform) | Between W3–W4 | F2–F8 |
| Heteroscedastic GP (LOO per-point alpha) | Between W4–W5 | F2 |

---

## Academic Foundations

The design choices in this project are grounded in established Bayesian Optimisation research. The table below maps each paper to the specific technique it motivates.

| Paper | Key idea applied in this project |
|---|---|
| Jones, Schonlau & Welch (1998). *Efficient Global Optimization of Expensive Black-Box Functions.* Journal of Global Optimization. | Expected Improvement (EI) acquisition function — used for F3, F5, F6, F7 where a credible incumbent exists and improvement focus is appropriate |
| Srinivas, Krause, Kakade & Seeger (2010). *Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design.* ICML. | GP-UCB acquisition with β parameter — provides theoretical regret bounds justifying the exploration pressure applied to F1, F2, F8 |
| Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning.* MIT Press. | Matérn 5/2 kernel selection; Automatic Relevance Determination (ARD) via per-dimension length-scales for F4, F7, F8 |
| Snoek, Larochelle & Adams (2012). *Practical Bayesian Optimization of Machine Learning Algorithms.* NeurIPS. | Motivation for normalising GP targets before fitting — basis for the output standardisation (`standardize` Y-transform) applied to F2–F8 |
| Kersting, Plagemann, Pfaff & Burgard (2007). *Most Likely Heteroscedastic Gaussian Process Regression.* ICML. | Input-dependent noise modelling — direct motivation for the per-point LOO alpha array used in F2's heteroscedastic GP |
| Lakshminarayanan, Pritzel & Blundell (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS. | Deep Ensemble uncertainty method — empirically tested in `analysis/05_nn_surrogate_analysis.ipynb` and found inferior to the GP at n ≤ 44 |
| Hutter, Hoos & Leyton-Brown (2011). *Sequential Model-Based Algorithm Configuration.* LION. | Random Forest surrogate with tree-variance uncertainty — evaluated in `analysis/03_function8_rf_surrogate.ipynb`; RF feature importance used to cross-validate GP ARD findings for F8 |
| Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR. | Core implementation library — `GaussianProcessRegressor` with native array alpha (heteroscedastic GP), ARD length-scales, and LOO cross-validation |

### Looking ahead

| Source | Relevance |
|---|---|
| Eriksson et al. (2019). *Scalable Global Optimization via Local Bayesian Optimization (TuRBO).* NeurIPS. | Trust region BO — constrains search to a local region around the current best; directly applicable to F8's 8D exploitation challenge |
| Balandat et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.* NeurIPS. | Production BBO library with Deep Kernel Learning support; worth adopting if dataset sizes grow beyond n ≈ 60 |
| Wilson, Hu, Salakhutdinov & Xing (2016). *Deep Kernel Learning.* AISTATS. | NN feature extractor + GP kernel — most promising NN-based surrogate architecture for higher-n settings; requires GPyTorch |

---

## Branch History

| Branch | Description |
|--------|-------------|
| `main` | Current working state |
| `week-1` | Snapshot after week 1 submissions |

---

# Contact

Justin Boynton <justin.boynton@gmail.com>
