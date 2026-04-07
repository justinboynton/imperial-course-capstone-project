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

**Candidate sampling:** 5,000 random uniform samples drawn per suggestion call; acquisition function evaluated on all candidates and the argmax returned.

---

## Function 1 — 2D Contamination / Radiation Field

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Localised hotspot likely creates a sharp, non-smooth peak. RBF oversmooths — changed at end of week 1. |
| **Y-transform** | `arcsinh` | Near-zero outputs dominate all observations (range ≈ 10⁻¹⁶ – 10⁻³). Standard GP fitting is unreliable at this scale. `arcsinh(Y/s)` is symmetric-log-like, defined for all reals, and spreads tiny magnitude differences so the GP can extract structure. Enabled week 3. |
| **Acquisition** | UCB | Pure exploration until a non-zero signal is found. |
| **β** | 2.0 | High — no informative non-zero signal found in three weeks. |
| **ξ** | 0.05 | Not active for UCB. |
| **Initial data** | 10 points, Y ≈ 0.000 everywhere | All near-zero. Hotspot not yet found. |

**Landscape notes:** Three weeks of zero returns across all four quadrants (top-right W1, lower-right W2, left-mid W3, initial cluster). The arcsinh transform gives the GP marginal structure from the near-zero spread but cannot compensate for a complete absence of signal. Manual grid exploration (lower-left [0.25, 0.25]) is the week 4 strategy.

**Changes log:**
- Week 1 end: Kernel `rbf → matern` (RBF inappropriate for sparse localised signals).
- Week 3: `arcsinh` Y-transform enabled to handle the 200-order-of-magnitude near-zero range.

---

## Function 2 — 2D Noisy Log-Likelihood

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Explicitly noisy with local optima — rougher surface. |
| **Acquisition** | UCB (W1–W3) → mean/EI ξ=0.001 (W4+) | UCB was used for exploration early. Week 3 returned Y=0.493, confirming the GP has located the right region. Switch to "mean" or very low ξ EI to exploit tightly. |
| **β** | 2.5 → 1.5 (reduce W4) | Reduce now that the peak neighbourhood is confirmed. |
| **ξ** | 0.1 → 0.001 (W4) | Tighten exploitation — near-greedy around [0.703, 0.927]. |
| **Initial data** | 10 points, Y ∈ [−0.066, 0.611] | Clear best at [0.703, 0.927] with Y=0.611. |

**Landscape notes:** W1 queried low-X₁ region (confirmed poor). W2 improved to 0.053. W3 targeted [0.694, 0.906] — large jump to 0.493. Gap to initial best (0.611) likely explained by function noise or a narrow peak requiring tighter search. Week 4 goal: beat 0.611.

**Changes log:**
- Week 3: Confirmed peak neighbourhood. Recommend switching acquisition to "mean" or EI ξ=0.001 for W4.

---

## Function 3 — 3D Drug Compound Combinations

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Physical/chemical process — smooth but not analytic. |
| **Acquisition** | EI | Focused improvement near known best. |
| **β** | 1.96 | Moderate. |
| **ξ** | 0.02 → 0.005 (W4) | Tighten — W3 catastrophic regression proved EI over-explored. |
| **Initial data** | 15 points, Y ∈ [−0.399, −0.035] | Best near [0.49, 0.61, 0.34]. |

**Landscape notes:** All outputs negative; maximising toward 0. W2 achieved −0.018 (best portal result). W3 sent Compound B → 0.030 — result collapsed to −0.123 (worst yet). **Hard constraint now enforced: Compound B ≥ 0.20.** Compound B near 0.30–0.60 is critical for good outcomes. W4 returns to [0.446, 0.339, 0.486] neighbourhood.

**Changes log:**
- Week 3: EI over-explored into harmful B < 0.10 region. ξ reduced to 0.005. Soft B-floor constraint (0.20) documented.

---

## Function 4 — 4D Warehouse ML Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Rough landscape with local optima. |
| **ARD** | Enabled | P3 shows no significant correlation (r=−0.16, p=0.38) while P1/P4 dominate (r≈−0.50). ARD assigns separate length-scales per dimension, effectively down-weighting P3 and allowing the GP to learn the anisotropic landscape correctly. |
| **Acquisition** | UCB → EI ξ=0.01 (W4) | Reduce exploration. W3 confirmed the peak is narrow — small deviations from W2 best caused regression. |
| **β** | 2.0 → 1.5 (W4) | Reduce — ARD now handles dimension weighting; aggressive exploration no longer warranted. |
| **ξ** | 0.05 → 0.01 (W4) | Tight exploitation near W2 best [0.460, 0.413, 0.311, 0.405]. |
| **Initial data** | 30 points, Y ∈ [−32.6, −4.0] | Best near [0.578, 0.429, 0.426, 0.249]. |

**Landscape notes:** Best portal result W2 (−1.177). W3 regressed to −1.568 despite staying close to W2 coordinates — confirms a narrow, sharp peak. P3 deviation (0.311 → 0.385) appears culpable. ARD should down-weight P3 length-scale as more data accumulates.

**Changes log:**
- Week 3: ARD Matérn enabled. Dimension-sensitivity analysis (r values) documented. Acquisition narrowing to EI ξ=0.01 for W4.

---

## Function 5 — 4D Chemical Yield (Unimodal)

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | RBF | Unimodal function — smooth differentiable surface. Confirmed by three consecutive improvements with tight exploitation. |
| **Acquisition** | EI (W1–W3) → **mean** (W4+) | "Mean" acquisition = pure GP posterior mean maximisation. No exploration bonus. Appropriate for unimodal functions once the peak region is confirmed. Enabled as a new option in the app. |
| **β** | 1.5 | Not relevant once switching to "mean". |
| **ξ** | 0.01 → 0.0 (W4, mean acq) | Not applicable for "mean" acquisition. |
| **Initial data** | 20 points, Y ∈ [0.113, 1088.86] | Outstanding best at [0.224, 0.846, 0.879, 0.879]. |

**Landscape notes:** Three consecutive portal improvements: 50.44 (W1) → 1138.87 (W2) → 1374.52 (W3, new all-time best). Unimodal structure confirmed with high confidence. C3 and C4 should stay ≥ 0.87; C1 has drifted upward (0.224 → 0.284 → 0.362 → target 0.38). W4 strategy: "mean" acquisition, no exploration.

**Changes log:**
- Week 2: EI tightened to ξ=0.01 after W1 wrong-direction result.
- Week 3: New all-time best (1374.52). "mean" acquisition added to app; switch planned for W4.

---

## Function 6 — 5D Cake Recipe (Negative Penalty)

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | Complex ingredient interactions — non-smooth. |
| **Acquisition** | EI | Exploit near known best. |
| **β** | 1.96 | Moderate. |
| **ξ** | 0.02 → 0.01 (W4) | Tighten exploitation — W3 achieved new best, converging. |
| **Initial data** | 20 points, Y ∈ [−2.571, −0.714] | Best at [0.728, 0.155, 0.733, 0.694, 0.056]. |

**Landscape notes:** Two consecutive portal improvements: −0.518 (W2) → −0.384 (W3, new all-time best). Now 46% better than initial best (−0.714). Early hypothesis (minimise Sugar and Milk) partially revised — moderate Sugar (0.48) still produced the best result. Emerging recipe profile: Flour≈0.38, Sugar≈0.26–0.48, Eggs≈0.44–0.56, Butter≈0.72, Milk≈0.16–0.17. W4 target: [0.37, 0.47, 0.57, 0.73, 0.17].

**Changes log:**
- Week 3: New all-time best (−0.384). ξ reducing to 0.01 for W4. Ingredient profile updated.

---

## Function 7 — 6D Gradient Boosting Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | ML hyperparameter surfaces are typically non-smooth. |
| **ARD** | Enabled | Dim1 (n_estimators) and Dim2 (learning_rate) dominate (inversely related, both sensitive). Dim5 (max_features) is less critical. ARD assigns short length-scales to sensitive dimensions and long ones to less-critical dims, improving acquisition landscape quality. |
| **Acquisition** | EI | Exploit near known best. |
| **β** | 1.96 | Moderate. |
| **ξ** | 0.05 → 0.01 (W4) | Reduce — W3 regression confirmed peak is sharp; over-exploration costly. |
| **Initial data** | 30 points, Y ∈ [0.003, 1.365] | Best at [0.058, 0.49, 0.25, 0.22, 0.42, 0.73]. |

**Landscape notes:** W2 achieved 2.358 (all-time best portal). W3 regressed to 1.931 — Dim2 (learning_rate) moved from 0.365 → 0.428 simultaneously with Dim1 0.095 → 0.061; this combination pushed outside the optimum. Dim6 (regularisation≈0.72) appears well-calibrated across all good results. W4 returns tightly to W2 best: [0.095, 0.365, 0.337, 0.317, 0.362, 0.721].

**Changes log:**
- Week 3: ARD Matérn enabled. Dim1/Dim2 sensitivity confirmed by W3 regression. ξ reducing to 0.01 for W4.

---

## Function 8 — 8D Complex ML Hyperparameters

| Setting | Value | Rationale |
|---------|-------|-----------|
| **Kernel** | Matérn 5/2 | High-dimensional hyperparameter surface — non-smooth. |
| **ARD** | Enabled | D1 (r=−0.65) and D3 (r=−0.68) have the strongest negative correlations with Y. RF feature importance analysis (notebook `03_function8_rf_surrogate.ipynb`) confirms D1 and D3 dominate. ARD assigns short length-scales to these critical dims and long scales to D5, D6, D8 which have weaker importance. |
| **Acquisition** | UCB → EI ξ=0.05 (W4) | W3 used β=3.5 (aggressive exploration) and severely regressed. Return to moderate EI for W4. |
| **β** | 3.5 (W3) → 2.0 (W4) | Reduce — aggressive exploration at β=3.5 was counterproductive. 8D space requires structured search, not random excursions. |
| **ξ** | 0.1 → 0.05 (W4) | Moderate — balance near W2 best with slight neighbourhood search. |
| **Initial data** | 40 points, Y ∈ [5.592, 9.599] | Best at params with low D1–D4, D6≈0.80, D8≈0.89. |

**Landscape notes:** W2 portal best (9.704) still all-time best. W3 catastrophic: β=3.5 sent D2=0.956, D4=0.908 alongside otherwise correct D1≈0.06, D3≈0.001. The combination of high D2 **and** high D4 severely penalised the output (7.318). Lesson: D1, D2, D3, D4 all need to be LOW simultaneously. Hard constraints for W4: D1 < 0.25, D2 < 0.30, D3 < 0.10, D4 < 0.10. W4 target: near W2 best [0.21, 0.20, 0.04, 0.04, 0.97, 0.07, 0.22, 0.06].

**Changes log:**
- Week 3: ARD Matérn enabled. Correlation analysis (D1 r=−0.65, D3 r=−0.68) and RF importance documented. β=3.5 exploration failed — β reduced to 2.0 and switching to EI for W4. D2/D4 must-be-LOW constraint established.
