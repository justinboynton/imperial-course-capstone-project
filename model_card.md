# Model Card — Surrogate Model Decisions

This document records the surrogate model configuration for each function and the rationale behind each choice. Updated when settings change.

---

## Surrogate Model Architecture

All functions use a **Gaussian Process Regressor** (scikit-learn `GaussianProcessRegressor`) as the surrogate model.

**Common settings:**
- `normalize_y=True` — handles scale differences across functions
- `alpha=1e-6` — small noise nugget for numerical stability
- `n_restarts_optimizer=5` — kernel hyperparameter optimisation restarts

**Training data:** Initial `.npy` observations (provided at challenge start) + all portal submissions, combined before each GP fit.

---

## Function 1 — 2D Contamination / Radiation Field

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Localised hotspot likely creates a sharp, non-smooth peak. RBF (original choice) oversmooths — changed at end of week 1. |
| **Acquisition** | UCB | Pure exploration required until hotspot is found. |
| **β** | 2.0 | High — no non-zero signal found yet. Reduce to 1.0 only after a meaningful Y is observed. |
| **ξ** | 0.05 | Not active for UCB. |
| **Initial data** | 10 points, Y ≈ 0.000 everywhere | All near-zero — confirms hotspot has not been hit yet. |

**Landscape notes:** Near-zero output everywhere in the initial data and week 1. The signal is almost certainly confined to a small localised region not yet sampled. Strategy is systematic quadrant-by-quadrant exploration.

**Changes log:**
- Week 1 end: Kernel changed `rbf → matern` (RBF inappropriate for sparse localised signals).

---

## Function 2 — 2D Noisy Log-Likelihood

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Explicitly noisy with local optima — rougher surface. |
| **Acquisition** | UCB → EI (recommended) | UCB used in week 1. AI analysis recommends switching to EI ξ=0.01 for exploitation near known best. |
| **β** | 2.5 | Currently high for exploration. Reduce once near [0.703, 0.927]. |
| **ξ** | 0.1 | Only relevant if EI/PI used. |
| **Initial data** | 10 points, Y ∈ [-0.066, 0.611] | Clear best at [0.703, 0.927] with Y=0.611. |

**Landscape notes:** Strong signal in high-X₁, high-X₂ quadrant. Week 1 queried low-X₁ region — confirmed poor. Next queries should converge toward [0.75, 0.93].

---

## Function 3 — 3D Drug Compound Combinations

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Physical/chemical process — smooth but not analytic. |
| **Acquisition** | EI | Focused improvement near known best. |
| **β** | 1.96 | Moderate. |
| **ξ** | 0.02 | Low — exploit near best. AI recommends reducing to 0.01. |
| **Initial data** | 15 points, Y ∈ [-0.399, -0.035] | Best near [0.49, 0.61, 0.34]. |

**Landscape notes:** All outputs negative (maximising negative side effects → closer to 0 is better). Week 1 moved to high-A, low-B region — worse result. Confirmed Compound B ≈ 0.6 is important.

---

## Function 4 — 4D Warehouse ML Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Rough landscape with local optima. |
| **Acquisition** | UCB → EI (recommended) | AI analysis recommends EI ξ=0.05. |
| **β** | 2.0 | Moderate exploration — dynamic environment. |
| **ξ** | 0.05 | Moderate — don't over-exploit in dynamic environment. |
| **Initial data** | 30 points, Y ∈ [-32.6, -4.0] | Best near [0.578, 0.429, 0.426, 0.249]. |

**Landscape notes:** All outputs negative. Week 1 used extreme values (P2≈0, P3≈1) — poor result. Best region is mid-range P1-P3, low P4. Dynamic environment means old observations may drift.

---

## Function 5 — 4D Chemical Yield (Unimodal)

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | RBF | Unimodal function — consistent with smooth, differentiable surface. Could switch to Matérn if peak not found. |
| **Acquisition** | EI | Focused exploitation near single peak. |
| **β** | 1.5 | Low — exploit known optimum neighbourhood. |
| **ξ** | 0.01 | Low — near-greedy exploitation. AI recommends reducing to 0.001. |
| **Initial data** | 20 points, Y ∈ [0.113, 1088.86] | Outstanding best at [0.224, 0.846, 0.879, 0.879]. |

**Landscape notes:** Huge gap between initial best (1088.86) and week 1 result (50.44). Week 1 moved to high-C1, low-C2 region — confirmed wrong direction. Target: low C1, high C2/C3/C4 near [0.22, 0.85, 0.88, 0.88].

---

## Function 6 — 5D Cake Recipe (Negative Penalty)

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Complex ingredient interactions — non-smooth. |
| **Acquisition** | EI | Exploit near known best. |
| **β** | 1.96 | Moderate. |
| **ξ** | 0.02 | Low-moderate. AI recommends reducing to 0.01. |
| **Initial data** | 20 points, Y ∈ [-2.571, -0.714] | Best at [0.728, 0.155, 0.733, 0.694, 0.056]. |

**Landscape notes:** All outputs negative — maximise toward 0. Best point has low Sugar (0.155) and low Milk (0.056). Week 1 used high Sugar (0.531) and high Milk (0.475) — confirmed penalising. Reduce Sugar < 0.2 and Milk < 0.1 in all future queries.

---

## Function 7 — 6D Gradient Boosting Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | ML hyperparameter surfaces are typically non-smooth. |
| **Acquisition** | EI | Exploit near known best. |
| **β** | 1.96 | Moderate. |
| **ξ** | 0.05 | Slightly higher — 6D space warrants some exploration. |
| **Initial data** | 30 points, Y ∈ [0.003, 1.365] | Best at [0.058, 0.49, 0.25, 0.22, 0.42, 0.73]. |

**Landscape notes:** Week 1 used informed start [0.333, 0.310, 0.250, 0.800, 0.800, 0.050] — opposite of best in key dimensions (high subsample/max_features vs low, low regularisation vs high). Next queries: perturb around [0.058, 0.49, 0.25, 0.22, 0.42, 0.73].

---

## Function 8 — 8D Complex ML Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | High-dimensional hyperparameter surface — non-smooth. |
| **Acquisition** | UCB | 8D space warrants maintained exploration. |
| **β** | 2.5 | Moderate-high — 8D GP is underdetermined. |
| **ξ** | 0.1 | Moderate — only relevant for EI/PI. |
| **Initial data** | 40 points, Y ∈ [5.592, 9.599] | Best at params with low P1-P4, P6≈0.80, P8≈0.89. |

**Landscape notes:** Week 1 result (9.26) close to but below initial best (9.60). Initial best not yet beaten. Pattern: very low P1-P4 with high P6, high P8 associated with best outcomes. Next queries: perturb initial best toward lower P1-P4.
