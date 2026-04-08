# Model Card — BBO Capstone Surrogate Models

Documents the surrogate model choices, acquisition settings, and learning outcomes for each of the eight black-box functions. Updated after each week's results are returned.

**Current status:** Week 3 results returned. Week 4 submissions pending.

---

## Global Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary surrogate** | Gaussian Process (GP) | Only common model providing both predictions and calibrated uncertainty — required for principled acquisition |
| **Default kernel** | Matérn 5/2 | Twice-differentiable; realistic for real-world processes that are smooth but not infinitely so |
| **Exception kernel** | RBF (F5 only) | Justified only for confirmed unimodal functions with smooth peaks |
| **Training data** | Initial `.npy` observations + all portal submissions | Integrated from Week 2 onward; Week 1 omitted initial data (bug, now fixed) |
| **Kernel stability** | Kernel fixed per function after initial choice | Kernel encodes a structural prior — changing it mid-run introduces inconsistency |
| **ARD** | Enabled selectively from Week 3 | Automatic Relevance Determination allows per-dimension lengthscales; useful for high-D functions once data accumulates |
| **Y-transform** | `standardize` (F2–F8), `arcsinh` (F1) | Z-scores Y before GP fitting so acquisition functions operate in consistent units; prevents outliers or large Y ranges from distorting kernel hyperparameter optimisation. Applied from Week 4 onward. `normalize_y=False` set on the GP to avoid double-normalisation. |

### Acquisition function guide

| Function | Formula | When used |
|----------|---------|-----------|
| **UCB** | `μ(x) + β·σ(x)` | Uncertain/noisy landscapes; β controls exploration pressure |
| **EI** | `E[max(f(x) − f*, 0)]` | When a clear incumbent exists; ξ controls required improvement margin |
| **PI** | `P[f(x) > f* + ξ]` | Conservative exploitation; not yet used in practice |
| **Mean** | `μ(x)` | Pure exploitation; used when the landscape is well-characterised (F5 Week 4) |

---

## Function 1 — 2D Contamination Field

**Dimensions:** 2 · **Initial data:** 10 points · **Output sign:** Near-zero (maximise toward a localised spike)

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Changed from RBF after Week 1 — RBF's smoothness assumption is wrong for a localised hotspot |
| Y-transform | arcsinh(Y) | Enabled from Week 3 — spreads the 200-order-of-magnitude output range for better GP fitting |
| Acquisition | UCB | No signal found — UCB's explicit exploration pressure is essential |
| β | 2.0 → 1.0 | β reduced to 1.0 in Week 4 pending; AI analysis suggests reducing once a weak signal was detected in W3 |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | best ≈ 0.000 | — | No meaningful signal in 10-point grid |
| 1 | [0.999, 0.986] | ≈ 1.5×10⁻¹⁸⁵ | UCB β=2.0, RBF | Top-right quadrant — zero |
| 2 | [0.856, 0.278] | ≈ −2.1×10⁻¹²² | UCB β=2.0, Matérn | Lower-right — zero |
| 3 | [0.150, 0.500] | ≈ 4.4×10⁻⁵⁷ | UCB β=2.0, Matérn | **Best so far** — left-centre, weak signal detected |
| 4 | [0.775, 0.763] | pending | UCB β=1.0, Matérn | — |

### All-time best

> ≈ 4.4×10⁻⁵⁷ — Week 3 — [0.150, 0.500]
> (Signal is real but extremely weak; hotspot not yet located)

### Key findings

- Three of four quadrants tested — only lower-left remains fully unexplored
- The GP cannot learn from near-zero data; arcsinh transform helps but is not a substitute for finding a real signal
- W3's marginal improvement suggests a very weak signal in the left half of the domain
- Week 4 strategy: systematic exploration of lower-left, targeting [0.25, 0.25]

### Exploratory analysis

None beyond the main GP — with effectively zero output everywhere, alternative surrogates (SVR, RF) have nothing to learn from.

---

## Function 2 — 2D Noisy Log-Likelihood

**Dimensions:** 2 · **Initial data:** 10 points · **Output range:** [−0.066, 0.611]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Handles noise and moderate roughness — confirmed appropriate |
| Acquisition | UCB | Function is noisy with local optima; UCB keeps mild exploration pressure |
| β | 2.5 (W2–W3) → 1.5 (W4) | Reduced in W4 to tighten exploitation around the confirmed peak |
| ξ | 0.1 throughout | Not relevant for UCB; retained in config |
| Y-transform | standardize | Z-scores targets so β/ξ have consistent cross-function meaning |
| Heteroscedastic | ✓ (W5) | Per-point noise via LOO residuals; peak region assigned ~1.6× more noise than flat region |

**Supplementary analysis:** `analysis/02_function2_svr_exploration.ipynb`
— SVR with RBF, polynomial and linear kernels compared against GP. Both GP and SVR-RBF agreed: next query should be near [0.70, 0.93]. SVR linear confirmed X₁ > 0.65 as the dominant direction.

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | 0.6112 at [0.703, 0.927] | — | Strong signal in high-X₁ / high-X₂ region |
| 1 | [0.116, 0.884] | 0.026 | UCB β=2.5, Matérn | Low-X₁ — confirmed bad region |
| 2 | [0.815, 0.962] | 0.053 | UCB β=2.5, Matérn | X₁ too high — peak is near 0.70, not 0.82 |
| 3 | [0.694, 0.906] | 0.493 | UCB β=2.5, Matérn | Close to initial best — 9× improvement on portal results |
| 4 | [0.700, 0.961] | pending | UCB β=1.5, Matérn, het-GP | — |

### All-time best

> 0.6112 — Initial data — [0.703, 0.927]
> Portal best: 0.493 (Week 3)

### Key findings

- Narrow peak around X₁ ≈ 0.70, X₂ ≈ 0.93 — moving X₁ rightward to 0.82 collapses the output
- Function is explicitly noisy — the gap between initial best (0.611) and W3 (0.493) at a distance of only 0.023 is partially observation noise, not a spatial gradient
- Heteroscedastic GP (W5): peak region near [0.70, 0.93] assigned alpha ≈ 1.47–1.58 vs alpha ≈ 0.94–1.18 in the flat region — prevents acquisition function from chasing noise-driven apparent gradients
- SVR and GP agree on the optimal region; additional models provide no new information
- Week 5 strategy: het-GP generates suggestions that sit within the genuine uncertainty cloud around [0.70, 0.93], rather than being deflected by the phantom 0.118 gradient

---

## Function 3 — 3D Drug Compound Combinations

**Dimensions:** 3 (A = Compound A, B = Compound B, C = Compound C) · **Initial data:** 15 points · **Output range:** [−0.399, −0.018]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Physical interaction model — moderate smoothness appropriate |
| Acquisition | EI | Clear incumbent exists from W2; EI focuses on beating it |
| ξ | 0.02 (W2–W3) → 0.05 (W4) | Slightly increased in W4 to avoid EI over-exploiting uncertain extremes (caused W3 regression) |
| β | 1.96 | Retained in config but not operative for EI |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | −0.0348 at [0.490, 0.612, 0.340] | — | Best initial: moderate A, mid-high B, low C |
| 1 | [0.779, 0.249, 0.419] | −0.042 | EI β=1.96, ξ=0.02, Matérn | High-A, low-B — worse; confirms B matters |
| 2 | [0.446, 0.339, 0.486] | **−0.018** | EI β=1.96, ξ=0.02, Matérn | **All-time best** — different region found |
| 3 | [0.559, 0.030, 0.554] | −0.123 | EI β=1.96, ξ=0.02, Matérn | B near zero — catastrophic regression |
| 4 | [0.983, 0.400, 0.609] | pending | EI β=1.96, ξ=0.05, Matérn | — |

### All-time best

> −0.0182 — Week 2 — [0.446, 0.339, 0.486]

### Key findings

- Compound B is the most sensitive dimension: B < 0.15 is catastrophic, B ∈ [0.30–0.65] is the safe zone
- The W2 best challenges early hypothesis (B ≈ 0.61 from initial data); B ≈ 0.34 produced better result
- EI with ξ=0.02 over-explored in W3 — sent B to 0.030 (5× away from known best B-value)
- Soft constraint enforced from W4: do not submit with B < 0.20
- Week 4 strategy: return to near W2 best [0.45, 0.34, 0.49], tighten ξ to 0.01

---

## Function 4 — 4D Warehouse ML Hyperparameters

**Dimensions:** 4 (P1–P4) · **Initial data:** 20 points · **Output range:** [−32.6, −1.177]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Rough landscape with large output swings |
| ARD | Enabled from W3 | Allows different lengthscales per dimension — P2 appears most sensitive |
| Acquisition | UCB | Noisy/dynamic environment; UCB handles uncertainty well |
| β | 2.0 (W2–W3) → 1.2 (W4) | Reducing β to tighten exploitation around W2 best |
| ξ | 0.05 throughout | Not operative for UCB |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | −4.026 at [0.578, 0.429, 0.426, 0.249] | — | Mid-range values across all dims |
| 1 | [0.438, 0.033, 0.982, 0.372] | −21.254 | UCB β=2.0, Matérn | Extreme values (P2≈0, P3≈1) — worst result |
| 2 | [0.460, 0.413, 0.311, 0.405] | **−1.177** | UCB β=2.0, Matérn | **All-time best** — 70.8% improvement |
| 3 | [0.456, 0.406, 0.385, 0.304] | −1.568 | UCB β=2.0, Matérn, ARD | Slight regression; P3 moved from 0.311 → 0.385 |
| 4 | [0.433, 0.397, 0.499, 0.420] | pending | UCB β=1.2, Matérn, ARD | — |

### All-time best

> −1.1765 — Week 2 — [0.460, 0.413, 0.311, 0.405]

### Key findings

- Mid-range values (0.31–0.46) dominate — extreme inputs produce catastrophic results
- P2 is the most sensitive dimension: P2 ≈ 0.03 caused a 5× regression in W1
- P3 = 0.311 outperformed P3 = 0.385 (W2 vs W3) — peak is sharp; ±0.07 deviation hurts
- ARD should distinguish P2's short lengthscale from the other dimensions
- Week 4 strategy: return close to W2 best [0.46, 0.41, 0.31, 0.41]

---

## Function 5 — 4D Chemical Yield

**Dimensions:** 4 (C1–C4) · **Initial data:** 20 points · **Output range:** [0.113, 1374.5]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | RBF | Confirmed unimodal — infinite differentiability assumption is justified |
| Acquisition | EI → Mean | EI ξ=0.01 for W2–W3; switched to pure GP mean maximisation (no exploration bonus) for W4 |
| ξ | 0.01 | Low exploitation threshold appropriate for confirmed unimodal landscape |
| β | 1.5 | Not operative for EI |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | 1088.86 at [0.224, 0.846, 0.879, 0.879] | — | Clear peak — low C1, high C2/C3/C4 |
| 1 | [0.817, 0.085, 0.387, 0.717] | 50.44 | EI β=1.5, ξ=0.01, RBF | High-C1, low-C2 — 21× below initial best |
| 2 | [0.284, 0.835, 0.910, 0.866] | 1138.87 | EI β=1.5, ξ=0.01, RBF | **New best** — +4.6% on initial best |
| 3 | [0.362, 0.837, 0.939, 0.872] | **1374.52** | EI β=1.5, ξ=0.01, RBF | **All-time best** — three consecutive improvements |
| 4 | [0.415, 0.859, 0.919, 0.797] | pending | Mean, RBF | — |

### All-time best

> 1374.524 — Week 3 — [0.362, 0.837, 0.939, 0.872]

### Key findings

- Unimodal structure confirmed: every query near the peak region improves; W1's far-field query failed completely
- C2 ≈ 0.836–0.846 and C4 ≈ 0.866–0.879 are tightly constrained — do not adjust
- C1 has drifted upward across weeks (0.224 → 0.362) while C3 has also drifted up (0.879 → 0.939); both trends rewarded
- RBF kernel is appropriate and confirmed by three consecutive improvements
- Week 4 strategy: pure mean maximisation; fix C2/C4, continue C1/C3 drift

---

## Function 6 — 5D Cake Recipe

**Dimensions:** 5 (Flour, Sugar, Eggs, Butter, Milk) · **Initial data:** 20 points · **Output range:** [−2.57, −0.384]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Food formulation — non-smooth interactions between ingredients |
| Acquisition | EI | Consistent improvement trend; EI guided three consecutive portal bests |
| ξ | 0.02 (W2–W3) → 0.05 (W4) | Slightly increased for W4 to allow the GP to follow the active gradient directions |
| β | 1.96 → 1.5 | Nominal; not operative for EI |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | −0.714 at [0.728, 0.155, 0.733, 0.694, 0.056] | — | Low Sugar, low Milk, high Flour/Eggs/Butter |
| 1 | [0.835, 0.531, 0.124, 0.512, 0.475] | −1.826 | EI β=1.96, ξ=0.02, Matérn | High Sugar, high Milk — strongly penalised |
| 2 | [0.446, 0.261, 0.435, 0.719, 0.162] | −0.518 | EI β=1.96, ξ=0.02, Matérn | **New best** (+27%) |
| 3 | [0.380, 0.481, 0.560, 0.725, 0.170] | **−0.384** | EI β=1.96, ξ=0.02, Matérn | **All-time best** — two consecutive portal improvements |
| 4 | [0.348, 0.501, 0.288, 0.969, 0.103] | pending | EI β=1.5, ξ=0.05, Matérn | — |

### All-time best

> −0.3837 — Week 3 — [0.380, 0.481, 0.560, 0.725, 0.170]

### Key findings

- Butter ≈ 0.72–0.73 is stable and load-bearing across all good results — do not move
- Milk must remain low (< 0.20) — every high-Milk query underperforms
- Counter-intuitive finding: moderate Sugar (0.26–0.48) outperforms extreme-low Sugar from initial data
- Flour is decreasing (0.728 → 0.380), Sugar and Eggs are increasing — the GP is following a genuine ridge
- Week 4 strategy: continue Flour↓, Sugar↑, Eggs↑ trajectory; anchor Butter and Milk

---

## Function 7 — 6D GBM Hyperparameters

**Dimensions:** 6 (D1=n_estimators, D2=learning_rate, D3=max_depth, D4=subsample, D5=max_features, D6=regularisation) · **Initial data:** 30 points · **Output range:** [0.003, 2.358]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | GBM hyperparameter landscapes are non-smooth with sharp interactions |
| ARD | Enabled from W3 | D1 and D6 appear to have short lengthscales (high sensitivity); ARD should reflect this |
| Acquisition | EI | Consistent improvement in W2; EI with moderate ξ has guided the best results |
| ξ | 0.05 (W2–W3) → 0.10 (W4) | Slightly increased in W4 — D4 and D5 remain under-explored |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | 1.365 at [0.058, 0.492, 0.247, 0.218, 0.420, 0.731] | — | Low n_est, mid LR, high regularisation |
| 1 | [0.333, 0.310, 0.250, 0.800, 0.800, 0.050] | 0.121 | EI β=1.96, ξ=0.05, Matérn | High subsample/max_features, low reg — opposite of best |
| 2 | [0.095, 0.365, 0.337, 0.317, 0.362, 0.721] | **2.358** | EI β=1.96, ξ=0.05, Matérn | **All-time best** — +72.6% on initial best |
| 3 | [0.061, 0.428, 0.284, 0.256, 0.374, 0.724] | 1.931 | EI β=1.96, ξ=0.05, Matérn, ARD | Regression — D2 moved too high (0.365 → 0.428) |
| 4 | [0.024, 0.110, 0.273, 0.338, 0.357, 0.537] | pending | EI β=1.96, ξ=0.10, Matérn, ARD | — |

### All-time best

> 2.3576 — Week 2 — [0.095, 0.365, 0.337, 0.317, 0.362, 0.721]

### Key findings

- Low D1 (n_estimators ≈ 0.06–0.10) + high D6 (regularisation ≈ 0.72) is the dominant axis
- W3 regression: D2 moved from 0.365 → 0.428 while D1 moved lower — the two interact; changing both simultaneously exited the optimal zone
- D6 ≈ 0.721–0.724 across the two best portal results — this is well-calibrated; do not disturb
- Week 4 strategy: return D2 to near 0.365, keep D1 low, hold D6 ≈ 0.72

---

## Function 8 — 8D ML Hyperparameters

**Dimensions:** 8 (D1–D8) · **Initial data:** 40 points · **Output range:** [5.59, 9.704]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | ML hyperparameter landscape — non-smooth with complex interactions |
| ARD | Enabled from W3 | 8D space; ARD critical for learning which dimensions are most influential |
| Acquisition | UCB | High-dimensional, sparse data — UCB's exploration bonus prevents premature convergence |
| β | 2.5 (W2) → 3.5 (W3, mistake) → 2.5 (W4) | W3's β=3.5 drove a catastrophic exploratory query; reverted |

**Supplementary analysis:** `analysis/03_function8_rf_surrogate.ipynb`
— Random Forest surrogate and GP+RF ensemble evaluated. RF provides permutation-based feature importance not available from GP alone. Key finding: D1 (r≈−0.65) and D3 (r≈−0.66) have strongest negative correlation with Y — must be kept LOW.

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Initial | — | 9.5985 at [0.056, 0.066, 0.023, 0.039, 0.404, 0.801, 0.488, 0.893] | — | Very low D1–D4, moderate D5, high D6/D8 |
| 1 | [0.242, 0.754, 0.171, 0.087, 0.351, 0.974, 0.195, 0.656] | 9.260 | UCB β=2.5, Matérn | Moved D2 high — near miss but below initial best |
| 2 | [0.212, 0.204, 0.040, 0.040, 0.973, 0.067, 0.219, 0.061] | **9.704** | UCB β=2.5, Matérn | **All-time best** — low D1–D4, high D5, low D6/D8 |
| 3 | [0.062, 0.956, 0.001, 0.908, 0.527, 0.002, 0.943, 0.228] | 7.318 | UCB β=3.5, Matérn, ARD | Aggressive exploration — D2=0.956, D4=0.908 simultaneously too high |
| 4 | [0.091, 0.141, 0.251, 0.059, 0.817, 0.769, 0.130, 0.089] | pending | UCB β=2.5, Matérn, ARD | — |

### All-time best

> 9.7035 — Week 2 — [0.212, 0.204, 0.040, 0.040, 0.973, 0.067, 0.219, 0.061]

### Key findings

- RF feature importance confirms: D1 and D3 are dominant negative contributors — keep < 0.25
- W3 failure: setting D2=0.956 and D4=0.908 simultaneously violated the low-D2-D4 constraint
- Hard constraints for remaining queries: D1 < 0.25, D2 < 0.30, D3 < 0.10, D4 < 0.10
- β=3.5 was too aggressive in W3 for an 8D function with only 43 total observations
- Week 4 strategy: exploit near W2 best [0.21, 0.20, 0.04, 0.04, 0.97, 0.07, 0.22, 0.06]

---

## Cross-Function Summary

| Fn | Dims | Kernel | Acquisition | β / ξ (current) | Y-transform | Initial Best Y | All-time Best Y | Best at Week |
|----|------|--------|-------------|-----------------|-------------|----------------|-----------------|--------------|
| 1 | 2 | Matérn 5/2 | UCB | β=1.0 | arcsinh | ≈0.000 | ≈4.4×10⁻⁵⁷ | W3 |
| 2 | 2 | Matérn 5/2 | UCB | β=1.5 | standardize | 0.6112 | 0.6112 | Initial |
| 3 | 3 | Matérn 5/2 | EI | ξ=0.05 | standardize | −0.0348 | **−0.0182** | W2 |
| 4 | 4 | Matérn 5/2 | UCB | β=1.2 | standardize | −4.026 | **−1.177** | W2 |
| 5 | 4 | RBF | Mean | ξ=0.01 | standardize | 1088.86 | **1374.52** | W3 |
| 6 | 5 | Matérn 5/2 | EI | ξ=0.05 | standardize | −0.714 | **−0.384** | W3 |
| 7 | 6 | Matérn 5/2 | EI | ξ=0.10 | standardize | 1.365 | **2.358** | W2 |
| 8 | 8 | Matérn 5/2 | UCB | β=2.5 | standardize | 9.5985 | **9.704** | W2 |

**Improved on initial best:** F3 ✓, F4 ✓, F5 ✓, F6 ✓, F7 ✓, F8 ✓ (6 of 8)
**Not yet improved:** F1 (no signal found), F2 (initial best is noisy; portal best is 0.493)

---

## Model Selection Rationale — Alternatives Considered

### Why not SVR as primary surrogate?

SVR via Support Vector Regression can model smooth functions well, as demonstrated in `analysis/02_function2_svr_exploration.ipynb` for Function 2. However, SVR provides no uncertainty estimate — making it unsuitable for principled acquisition functions (UCB, EI, PI all require a posterior variance). It is used as a validation cross-check only.

### Why not Random Forest as primary surrogate?

Random Forest was evaluated as a SMAC-style surrogate for Function 8 in `analysis/03_function8_rf_surrogate.ipynb`. RF provides useful feature importance not available from the GP, and tree-variance can proxy uncertainty. However, RF uncertainty estimates are poorly calibrated compared to GP posteriors. The GP+RF ensemble (RF models the global trend, GP models residuals) is theoretically stronger but computationally complex for a weekly-cadence challenge. RF is used as a diagnostic tool to validate GP-suggested dimensions.

### Why not a neural network?

With 13–43 observations per function, a neural network would overfit severely. Additionally, tuning a neural network's own hyperparameters would require a secondary BBO process. The GP's analytical hyperparameter fitting (maximum likelihood) avoids this bootstrapping problem entirely.

---

## Engineering Changes Log

### W4 — Output standardisation (`"standardize"` Y-transform)

**Implemented:** 2026-04-08, between W3 results and W4 submissions.

**Change:** Added a `"standardize"` Y-transform to `capstone_app.py`. Applied to F2–F8 via `FUNCTION_CONFIG`. F1 retains its existing `arcsinh` transform. The GP is built with `normalize_y=False` whenever any Y-transform is active.

**Pipeline (per suggestion call):**
```
Y_raw  →  Y_fit = (Y_raw − μ) / σ  →  GP.fit(X, Y_fit)
GP.predict(x_candidate) → mean_z, std_z   [in z-score units]
UCB / EI computed on mean_z, std_z
mean_display = mean_z × σ + μ             [reverted for UI]
std_display  = std_z  × σ                 [reverted for UI]
```

**Root cause addressed:** scikit-learn's `GaussianProcessRegressor.predict` returns values in the same scale passed to `fit`. Before this change, the acquisition function (UCB/EI) operated in raw Y units, meaning β and ξ had implicitly different meanings across the eight functions:

| Fn | Raw σ(Y) | EI ξ=0.01 meaning (before) | EI ξ=0.01 meaning (after) |
|----|----------|---------------------------|--------------------------|
| F3 | ≈ 0.08 | improve by 12.5% of σ | improve by 1% of σ |
| F4 | ≈ 8.3 | improve by 0.1% of σ | improve by 1% of σ |
| F5 | ≈ 577 | improve by 0.002% of σ | improve by 1% of σ |
| F7 | ≈ 0.8 | improve by 1.25% of σ | improve by 1% of σ |

After standardisation, ξ=0.01 uniformly means "require improvement of 1% of one standard deviation above the current best" — a consistent, interpretable threshold.

**Behavioural impact:**
- UCB suggestions: unaffected in direction (linear rescaling of mean and std does not change the argmax)
- EI suggestions: effective ξ threshold shifts. For F4 and F5 (large raw σ) the effective threshold decreases, making EI more exploitation-focused — appropriate given these functions are in pure exploit mode
- Visualisation: unchanged — dashboard charts still use raw Y throughout

---

### W5 — Heteroscedastic GP for F2 (`"heteroscedastic": True`)

**Implemented:** 2026-04-08, between W4 submission and W5.

**Applies to:** Function 2 only.

**Root cause addressed:** Two nearby observations near the known F2 peak — `[0.703, 0.927] → 0.611` (initial data) and `[0.694, 0.906] → 0.493` (W3) — are only 0.023 apart spatially but 0.118 apart in output. A homoscedastic GP (constant `alpha = 1e-6`) interprets this gap as a genuine spatial gradient. The acquisition function then pushes subsequent queries away from the true peak toward what appears to be the "uphill" side, when in reality the gap is largely observation noise.

**Change:** Added `compute_heteroscedastic_alpha(X, Y_fit)` to `capstone_app.py`. When `cfg["heteroscedastic"]` is `True`, this function runs before GP fitting and returns a per-point `alpha` array.

**Algorithm:**

```
For each training point i (n=14 for F2):
    Fit GP on remaining n-1 points
    Predict Y_fit[i] → get LOO residual r_i = (Y_fit[i] - pred_i)²

For each i:
    alpha_i = Σ_j [ exp(-‖x_i - x_j‖² / 2h²) · r_j ] / Σ_j weights
    (Gaussian kernel smoother, h = 0.20)

Clip alpha_i ≥ 1e-4
```

**Per-point alpha on F2 data (n=14, Y_fit = standardised):**

| Region | alpha |
|---|---|
| Peak [0.703, 0.927] | 1.487 |
| Peak [0.694, 0.906] | 1.466 |
| Flat [0.143, 0.349] | 1.182 |
| Flat [0.339, 0.214] | 0.936 |

Peak region carries **1.6× higher noise estimate** than the flat region.

**Implementation notes:**
- `normalize_y=False` used with the per-point alpha (Y_fit already z-scored; alpha values are in z-score² units — consistent)
- `_prepare_gp` (visualisation) also uses het-GP, converting alpha to normalised units via `alpha_norm = alpha_raw / Y.std()²` since visualisation GPs use raw Y with `normalize_y=True`
- Falls back to constant `alpha = 1e-4` when n < 4 (LOO unreliable with very few points)
- Dashboard: purple `het-GP` badge shown alongside `standardize` badge for F2

**Behavioural impact:**
- UCB suggestions for F2 will now sit within the genuine uncertainty cloud around [0.70, 0.93], rather than being deflected away by the phantom 0.118 gradient between the two nearby peak observations
- CI bands in the F2 GP slice plots will be noticeably wider in the peak region — an honest representation of our uncertainty about the true maximum location
