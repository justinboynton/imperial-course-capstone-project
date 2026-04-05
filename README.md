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

### Career relevance

BBO and Bayesian Optimisation are directly applicable to:

- **ML Engineering** — automated hyperparameter tuning is standard practice; understanding BO makes you a more effective engineer who can go beyond grid search and random search
- **MLOps** — Bayesian optimisation underlies tools like Optuna, Ray Tune and Google Vizier; understanding its internals helps when those tools need to be configured or extended
- **Research** — the skill of reasoning about an unknown objective from limited evidence, balancing exploration and exploitation, and iterating a strategy based on outcomes is transferable to any empirical research setting
- **Data Science consulting** — clients often need optimisation under real-world constraints (cost, time, number of experiments); BBO provides a principled framework to frame and solve these problems

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

Output characteristics vary significantly by function:

| Function | Output range (observed) | Sign | Domain context |
|----------|------------------------|------|----------------|
| F1 | ≈ [−0.004, 0] | Negative/zero | Contamination signal — very small values everywhere except near the source |
| F2 | [−0.066, 0.611] | Mixed | Noisy log-likelihood |
| F3 | [−0.399, −0.035] | Negative | Negative side-effect count; maximise toward 0 |
| F4 | [−32.6, −4.0] | Negative | Difference from baseline; maximise toward 0 |
| F5 | [0.113, 1088.9] | Positive | Chemical yield — unimodal, one clear peak |
| F6 | [−2.57, −0.71] | Negative | Negative penalty score; maximise toward 0 |
| F7 | [0.003, 1.365] | Positive | ML model performance score |
| F8 | [5.59, 9.60] | Positive | Validation accuracy (scaled) |

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
| **Local optima risk** | Some functions have multiple peaks (e.g. F1's contamination model may have two sources); finding a local maximum may not be the global one |

### What "success" looks like

A successful submission is not necessarily the global maximum — it is **a higher output than all previous submissions for that function**. Progress is measured week-on-week, and the capstone rewards thoughtful iteration and documented reasoning as much as raw performance.

---

## Section 4: Technical Approach

### Core framework: Gaussian Process Bayesian Optimisation

The primary strategy throughout this project is **Bayesian Optimisation (BO)** using a **Gaussian Process (GP) surrogate model**. The GP is fitted on all available observations (provided initial data + all weekly portal submissions combined) and used to:

1. **Predict the expected output** at any unsampled point (posterior mean)
2. **Quantify uncertainty** at any unsampled point (posterior standard deviation)
3. **Guide the next query** via an acquisition function that balances exploration and exploitation

This approach is implemented in `capstone_app.py` (Streamlit dashboard) and supports per-function configuration of kernel, acquisition function and hyperparameters.

### Acquisition functions used

| Function | Strategy | Rationale |
|----------|----------|-----------|
| **Upper Confidence Bound (UCB)** | `μ(x) + β·σ(x)` | Explicit exploration–exploitation trade-off via β; favoured for functions where the landscape is uncertain (F1, F2, F4, F8) |
| **Expected Improvement (EI)** | Probability-weighted gain over current best | Concentrates queries near the current best; favoured for unimodal or near-exploitation settings (F3, F5, F6, F7) |
| **Probability of Improvement (PI)** | Probability of exceeding best + ξ | Available as a supplementary strategy; useful when a conservative improvement is acceptable |

### Kernel selection

The kernel defines the assumed smoothness of the unknown function:

| Kernel | When used | Why |
|--------|----------|-----|
| **Matérn 5/2** | Most functions (F2–F4, F6–F8) | Allows moderate roughness; robust assumption for real-world functions that are smooth but not infinitely so |
| **RBF** | F5 (chemical yield) | The function is described as unimodal; RBF's infinite smoothness assumption is appropriate when a single peak exists |
| **Matérn 5/2** | F1 (contamination) | Switched from RBF after week 1 — the near-zero output everywhere except near the source suggests a sharp, localised response that Matérn handles better |

Kernels are kept **fixed per function throughout the competition** once set. Changing the kernel mid-run introduces inconsistency in the GP's beliefs about the function landscape without strong evidence justifying the change.

### Balancing exploration and exploitation

The exploration–exploitation trade-off is managed at two levels:

**1. Acquisition function choice**
- EI is used when the GP has a credible estimate of the peak location (exploit)
- UCB with higher β is used when the landscape remains uncertain (explore)

**2. Per-function β and ξ tuning**
- Lower β (≈1.5–2.0): more exploitation-focused
- Higher β (≈2.5–3.0): more exploration-focused
- ξ (EI/PI offset): kept small (0.01–0.05) to encourage incremental improvement near the current best

**3. Manual overrides**
For F1 — where the GP has found almost no signal — pure GP guidance is overridden with a systematic quadrant-search strategy. The 10 initial points cluster in the top-right quadrant. Weeks 2 onward explore each of the three remaining quadrants in turn.

### Why not SVMs or logistic regression?

Both were considered as alternative surrogate approaches. SVMs via Support Vector Regression (SVR) can model smooth functions, but provide no uncertainty estimate — making them unsuitable for principled exploration. Logistic regression is a classification method and is not applicable to continuous output regression. The GP remains the correct tool here because it is the only commonly available model that simultaneously provides **predictions and calibrated uncertainty estimates** over a continuous domain.

### AI-assisted strategy analysis

The dashboard integrates **Anthropic Claude** (claude-sonnet-4-6) to generate per-function strategy analysis. At each stage, Claude receives: function metadata, all initial data statistics, all portal observations so far, and the current best result. It returns a structured analysis covering landscape assessment, acquisition strategy review, a recommended next query and overall strategic direction. These analyses are stored persistently in `capstone_history.json` and viewable via the app's AI tab.

### Exploratory analysis

An EDA notebook (`analysis/01_initial_data_eda.ipynb`) was produced in week 1 to document the initial data landscape before any portal submissions. Key findings:

- All 8 week 1 submissions landed below the initial best — caused by the initial `.npy` data not being integrated into the GP at query time. Corrected from week 2 onward.
- F5 has the largest opportunity — initial best Y=1088.86, week 1 returned 50.44. Unimodal structure makes this highly exploitable.
- F1 is the hardest — near-zero output everywhere; hotspot not yet located.
- Over-exploration was the dominant week 1 error across all 8 functions.

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
├── initial_data/                # Provided initial .npy observations
│   ├── function_1/ ... function_8/
├── analysis/                    # Exploratory analysis notebooks and charts
│   └── 01_initial_data_eda.ipynb
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

---

# Contact

Justin Boynton <justin.boynton@gmail.com>
