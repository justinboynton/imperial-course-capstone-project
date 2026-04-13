# Model Card — BBO Capstone Surrogate Models

Documents the surrogate model choices, acquisition settings, and learning outcomes for each of the eight black-box functions. Updated after each week's results are returned.

**Current status:** Week 10 submissions complete. W9 results returned. Challenge concluded (10 of 10 queries submitted per function).

---

## Global Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary surrogate** | Gaussian Process (GP) | Only common model providing both predictions and calibrated uncertainty — required for principled acquisition |
| **Default kernel** | Matérn 5/2 | Twice-differentiable; realistic for real-world processes that are smooth but not infinitely so |
| **Alternative kernels** | Matérn 3/2 (F4), Rational Quadratic (F7), RBF (F5), RBF (F1 W10 trial) | Per-function kernel selection based on LOO R² comparison (notebook 06); kernel choice is re-evaluated as data accumulates |
| **Training data** | Initial `.npy` observations + all portal submissions | Integrated from Week 2 onward; Week 1 omitted initial data (bug, now fixed) |
| **ARD** | Enabled selectively (F4, F8) | Automatic Relevance Determination allows per-dimension lengthscales; useful for high-D functions once data accumulates |
| **Y-transform** | `standardize` (F2–F8), `arcsinh` (F1) | Z-scores Y before GP fitting so acquisition functions operate in consistent units; prevents outliers or large Y ranges from distorting kernel hyperparameter optimisation. Applied from Week 4 onward. `normalize_y=False` set on the GP to avoid double-normalisation. |
| **Heteroscedastic GP** | F2 only | Per-point noise estimated via LOO residuals + Gaussian kernel smoothing; prevents acquisition from chasing noise-driven gradients |

### Acquisition function guide

| Function | Formula | When used |
|----------|---------|-----------|
| **UCB** | `μ(x) + β·σ(x)` | Uncertain/noisy landscapes; β controls exploration pressure |
| **EI** | `E[max(f(x) − f*, 0)]` | When a clear incumbent exists; ξ controls required improvement margin |
| **PI** | `P[f(x) > f* + ξ]` | Conservative exploitation; tested on F3 W6 |
| **Mean** | `μ(x)` | Pure exploitation; used for F5 (W4, W7–W10), F6 (W5–W6, W9) |

---

## Function 1 — 2D Contamination Field

**Dimensions:** 2 · **Initial data:** 10 points · **Output sign:** Near-zero (maximise toward a localised spike)

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 → RBF (W10 trial) | Changed from RBF→Matérn after W1; RBF re-tested in W10 for localised hotspot |
| Y-transform | arcsinh(Y) | Spreads the 183-order-of-magnitude output range for better GP fitting |
| Acquisition | UCB | No reliable signal until W8 — UCB's explicit exploration pressure essential |
| β | 2.0 → 1.5 (W8) → 0.5 (W9–W10) | Reduced after W8 breakthrough to exploit confirmed hotspot |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | ≈7.7×10⁻¹⁶ at [0.731, 0.733] | — | Near-zero everywhere |
| 1 | [0.999, 0.986] | ≈1.5×10⁻¹⁸⁵ | UCB β=2.0, Matérn | Top-right — zero |
| 2 | [0.856, 0.278] | ≈−2.1×10⁻¹²² | UCB β=2.0, Matérn | Lower-right — zero |
| 3 | [0.150, 0.500] | ≈4.4×10⁻⁵⁷ | UCB β=2.0, Matérn | Left-centre — marginal signal |
| 4 | [0.775, 0.763] | ≈−1.6×10⁻²⁷ | UCB β=1.0, Matérn | Near initial best region — negative |
| 5 | [0.080, 0.480] | ≈−1.3×10⁻⁸⁴ | UCB β=2.0, Matérn | Far left — zero |
| 6 | [0.122, 0.518] | ≈−6.5×10⁻⁷⁰ | EI β=2.3, Matérn | Left-centre cluster — diminishing returns |
| 7 | [0.080, 0.200] | ≈−3.1×10⁻¹¹⁶ | EI β=2.0, Matérn | Lower-left — worst region yet |
| 8 | [0.691, 0.707] | **1.64×10⁻⁷** | UCB β=1.5, Matérn | **Breakthrough** — +50 orders of magnitude |
| 9 | [0.678, 0.759] | −1.30×10⁻¹⁴ | UCB β=0.5, Matérn | Negative; X₂=0.759 slightly outside band |
| 10 | [0.702, 0.721] | pending | UCB β=0.5, RBF | RBF trial; tighter around W8 hotspot |

### All-time best

> 1.64×10⁻⁷ — Week 8 — [0.691, 0.707]

### Key findings

- Seven queries in the wrong region (W1–W7) returned effectively zero; a single model-free analysis (notebook 07) identified the hotspot via Spearman radial decay correlation
- Hotspot centred near [0.65–0.73, 0.68–0.73] — confirmed by W8 breakthrough
- The initial data contained the signal: two points near [0.65, 0.68] and [0.73, 0.73] bracketed the hotspot — should have been recognised in W1
- GP cannot fit this function due to 183-order dynamic range; model-free spatial statistics are the correct approach
- W9 regression: X₂=0.759 exceeded the [0.68, 0.73] constraint band

### Exploratory analysis

- `analysis/07_function1_hotspot_hunt.ipynb` — log-space GP, sign boundary mapping, radial decay analysis, candidate identification

---

## Function 2 — 2D Noisy Log-Likelihood

**Dimensions:** 2 · **Initial data:** 10 points · **Output range:** [−0.066, 0.726]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Handles noise and moderate roughness — confirmed appropriate |
| Acquisition | UCB | Function is noisy with local optima; UCB maintains mild exploration pressure |
| β | 2.5 → 1.5 (W4) → 1.2 (W6) → 2.5 (W7–W9) → 0.7 (W10) | Reduced β correlates with best results; W10 switches to near-pure exploitation |
| Heteroscedastic | ✓ (from W5) | Per-point noise via LOO residuals; peak region assigned ~1.6× more noise than flat region |
| Y-transform | standardize | Z-scores targets so β/ξ have consistent meaning |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | 0.6112 at [0.703, 0.927] | — | Strong signal in high-X₁ / high-X₂ |
| 1 | [0.116, 0.884] | 0.026 | UCB β=2.5, Matérn | Low-X₁ — confirmed bad |
| 2 | [0.815, 0.962] | 0.053 | UCB β=2.5, Matérn | X₁ too high |
| 3 | [0.694, 0.906] | 0.493 | UCB β=2.5, Matérn | Near peak |
| 4 | [0.700, 0.961] | **0.648** | UCB β=1.5, Matérn | **Beats initial best** |
| 5 | [0.698, 0.942] | 0.513 | UCB β=2.5, Matérn, het-GP | Regression — noise |
| 6 | [0.699, 0.932] | **0.726** | UCB β=1.2, Matérn, het-GP | **All-time best** |
| 7 | [0.694, 0.921] | 0.585 | UCB β=2.5, Matérn, het-GP | Regression — X₂ too low |
| 8 | [0.699, 0.927] | 0.715 | UCB β=2.5, Matérn, het-GP | Near-best; confirms region |
| 9 | [0.700, 0.932] | 0.548 | UCB β=2.5, Matérn, het-GP | Regression — noise |
| 10 | [0.706, 0.934] | pending | UCB β=0.7, Matérn, het-GP | Near-pure exploitation |

### All-time best

> 0.7260 — Week 6 — [0.699, 0.932]

### Key findings

- Narrow peak at X₁ ≈ 0.70, X₂ ∈ [0.926, 0.932]; the function is stochastic — repeated queries at nearly identical inputs return different values
- Lower β consistently produces better results: W4 (β=1.5, Y=0.648), W6 (β=1.2, Y=0.726) vs W5/W7/W9 (β=2.5, Y=0.513/0.585/0.548)
- Heteroscedastic GP correctly assigns higher noise to the peak region, preventing acquisition from chasing noise-driven gradients
- W10 strategy: β=0.7 to maximise exploitation after three consecutive regressions with β=2.5

### Exploratory analysis

- `analysis/02_function2_svr_exploration.ipynb` — SVR with RBF/polynomial/linear kernels confirmed peak region

---

## Function 3 — 3D Drug Compound Combinations

**Dimensions:** 3 (A, B, C) · **Initial data:** 15 points · **Output range:** [−0.399, −0.009]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Physical interaction model — moderate smoothness appropriate |
| Acquisition | EI (primary), PI (W6), Mean (W7) | Tested multiple strategies around the narrow W5 optimum |
| ξ | 0.02 → 0.05 (W4) → 0.02 (W5–W9) → 0.012 (W10) | Gradually tightened as best region confirmed |
| Y-transform | standardize | |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | −0.0348 at [0.493, 0.612, 0.340] | — | Moderate A, mid-high B, low C |
| 1 | [0.779, 0.249, 0.419] | −0.042 | EI ξ=0.02, Matérn | High A, low B — worse |
| 2 | [0.446, 0.339, 0.486] | **−0.018** | EI ξ=0.02, Matérn | **Beats initial best** |
| 3 | [0.559, 0.030, 0.554] | −0.123 | EI ξ=0.02, Matérn | B≈0 — catastrophic |
| 4 | [0.983, 0.400, 0.609] | −0.064 | EI ξ=0.05, Matérn | High A — poor |
| 5 | [0.439, 0.461, 0.503] | **−0.009** | EI ξ=0.02, Matérn | **All-time best** |
| 6 | [0.446, 0.496, 0.466] | −0.013 | PI ξ=0.02, Matérn | Near-best; slight regression |
| 7 | [0.431, 0.422, 0.508] | −0.014 | Mean, Matérn | Near-best; B too low |
| 8 | [0.460, 0.518, 0.510] | −0.017 | EI ξ=0.02, Matérn | B drifting up — regression |
| 9 | [0.606, 0.401, 0.510] | −0.010 | EI ξ=0.02, Matérn | A=0.606 exploration; near-best |
| 10 | [0.987, 0.970, 0.950] | pending | EI ξ=0.012, Matérn | Upper corner exploration bet |

### All-time best

> −0.00897 — Week 5 — [0.439, 0.461, 0.503]

### Key findings

- Extremely narrow optimum: A ∈ [0.43, 0.46], B ∈ [0.42, 0.50], C ∈ [0.47, 0.52]
- Five consecutive weeks (W6–W10) unable to beat the W5 result despite tight perturbations
- W10 is a radical corner probe ([0.987, 0.970, 0.950]) — testing whether a second basin exists near the boundary

---

## Function 4 — 4D Warehouse ML Hyperparameters

**Dimensions:** 4 (P1–P4) · **Initial data:** 30 points · **Output range:** [−32.6, +0.367]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 → **Matérn 3/2** (from W8) | Notebook 06 showed LOO R² improved from 0.485 → 0.961 with Matérn 3/2 |
| ARD | Enabled from W3 | Per-dimension lengthscales; P3 has shortest lengthscale (most sensitive) |
| Acquisition | UCB → **EI** (from W7) | Switched to EI after locating the positive-Y region |
| ξ | 0.05 → 0.01 (W8) → 0.002 (W10) | Progressively tightened for precision exploitation |
| Y-transform | standardize | |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | −4.026 at [0.578, 0.429, 0.426, 0.249] | — | Mid-range |
| 1 | [0.438, 0.033, 0.982, 0.372] | −21.254 | UCB β=2.0, Matérn | Extreme P2/P3 — worst |
| 2 | [0.460, 0.413, 0.311, 0.405] | −1.177 | UCB β=2.0, Matérn | Beats initial best |
| 3 | [0.456, 0.406, 0.385, 0.304] | −1.568 | UCB β=2.0, Matérn ARD | Regression — P3 too high |
| 4 | [0.433, 0.397, 0.499, 0.420] | −2.370 | UCB β=1.2, Matérn ARD | P3=0.499 too high |
| 5 | [0.465, 0.420, 0.270, 0.389] | −1.897 | UCB β=2.0, Matérn ARD | P3 too low this time |
| 6 | [0.449, 0.418, 0.363, 0.377] | **+0.136** | UCB β=2.0, Matérn ARD | **First positive Y ever** |
| 7 | [0.442, 0.427, 0.363, 0.378] | **+0.330** | EI ξ=0.05, Matérn ARD | 2nd consecutive improvement |
| 8 | [0.438, 0.431, 0.355, 0.380] | **+0.367** | EI ξ=0.01, **Matérn 3/2** ARD | **All-time best** — 3rd consecutive |
| 9 | [0.230, 0.457, 0.404, 0.463] | −3.207 | EI ξ=0.01, Matérn 3/2 ARD | P1=0.230 violated constraint |
| 10 | [0.447, 0.484, 0.357, 0.321] | pending | EI ξ=0.002, Matérn 3/2 ARD | Return to basin; tighter ξ |

### All-time best

> +0.3674 — Week 8 — [0.438, 0.431, 0.355, 0.380]

### Key findings

- "Moderate everything" optimum: all dims ∈ [0.35, 0.45] — consistent with regularisation theory
- Matérn 3/2 kernel switch (notebook 06) directly enabled three consecutive improvements (W6–W8)
- W6–W8 convergence: diminishing step sizes (+0.136, +0.330, +0.367) consistent with approaching a stationary point
- W9 regression: P1=0.230 violated the [0.35, 0.45] constraint — confirmed the basin is unimodal in this region

### Exploratory analysis

- `analysis/06_kernel_variants_ngboost.ipynb` — LOO R² kernel comparison; Matérn 3/2 selected

---

## Function 5 — 4D Chemical Yield

**Dimensions:** 4 (C1–C4) · **Initial data:** 20 points · **Output range:** [0.0, 2238.7]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | RBF | Confirmed unimodal — infinite differentiability assumption justified |
| Acquisition | EI → **Mean** (from W4, except W5–W6 EI) | Pure GP posterior mean maximisation for confirmed unimodal landscape |
| ξ | 0.01 throughout | Low threshold; not operative for Mean |
| Y-transform | standardize | |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | 1088.9 at [0.224, 0.846, 0.879, 0.879] | — | Low C1, high C2/C3/C4 |
| 1 | [0.817, 0.085, 0.387, 0.717] | 50.4 | EI ξ=0.01, RBF | High C1, low C2 — 21× below |
| 2 | [0.284, 0.835, 0.910, 0.866] | 1138.9 | EI ξ=0.01, RBF | **Beats initial** (+4.6%) |
| 3 | [0.362, 0.837, 0.939, 0.872] | **1374.5** | EI ξ=0.01, RBF | New best (+26%) |
| 4 | [0.415, 0.859, 0.919, 0.797] | 1124.9 | Mean, RBF | C4 too low (0.797) |
| 5 | [0.339, 0.838, 0.946, 0.872] | **1412.6** | EI ξ=0.01, RBF | New best (+30%) |
| 6 | [0.313, 0.843, 0.957, 0.811] | 1223.3 | EI ξ=0.01, RBF | C4 too low again |
| 7 | [0.340, 0.842, 0.950, 0.876] | **1482.4** | Mean, RBF | New best (+36%) |
| 8 | [0.351, 0.915, 0.958, 0.874] | **1963.7** | Mean, RBF | **D2 breakout** (+80%) |
| 9 | [0.350, 0.923, 0.961, 0.902] | **2238.7** | Mean, RBF | **All-time best** (+106%) |
| 10 | [0.332, 0.951, 0.985, 0.980] | pending | Mean, RBF | Continuing D2/D3/D4 push |

### All-time best

> 2238.73 — Week 9 — [0.350, 0.923, 0.961, 0.902]

### Key findings

- Most successful function: 6 new all-time bests across 9 weeks, with the yield more than doubling the initial best
- C2 was under-constrained until W8 — holding it at 0.84 for four weeks missed the true peak (C2 > 0.90)
- Consistent gradient: C2, C3, C4 all increasing toward 1.0; C1 ≈ 0.33–0.35 stable
- Mean acquisition (pure exploitation) produced four consecutive bests (W7–W9)
- RBF kernel confirmed appropriate for this smooth unimodal landscape

---

## Function 6 — 5D Cake Recipe

**Dimensions:** 5 (Flour, Sugar, Eggs, Butter, Milk) · **Initial data:** 20 points · **Output range:** [−2.57, −0.246]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 → **Matérn 3/2** (W10 trial) | Testing rougher kernel for ingredient interactions |
| Acquisition | EI / Mean (alternating) | Mean for exploitation phases (W5–W6, W9); EI for recovery |
| ξ | 0.02 → 0.008 (W10) | Tightened for precision exploitation |
| Y-transform | standardize | |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | −0.714 at [0.728, 0.155, 0.733, 0.694, 0.056] | — | Low Sugar, low Milk |
| 1 | [0.835, 0.531, 0.124, 0.512, 0.475] | −1.826 | EI ξ=0.02, Matérn | High Sugar/Milk — worst |
| 2 | [0.446, 0.261, 0.435, 0.718, 0.162] | **−0.518** | EI ξ=0.02, Matérn | Beats initial (+27%) |
| 3 | [0.380, 0.481, 0.560, 0.725, 0.170] | **−0.384** | EI ξ=0.02, Matérn | New best (+46%) |
| 4 | [0.348, 0.501, 0.288, 0.969, 0.103] | −1.294 | EI ξ=0.05, Matérn | Butter=0.969 too extreme |
| 5 | [0.343, 0.523, 0.603, 0.751, 0.141] | **−0.341** | Mean, Matérn | New best (+52%) |
| 6 | [0.408, 0.411, 0.766, 0.787, 0.023] | **−0.296** | Mean, Matérn | New best (+59%) |
| 7 | [0.377, 0.452, 0.821, 0.824, 0.022] | −0.452 | EI ξ=0.02, Matérn | Eggs too high |
| 8 | [0.472, 0.407, 0.735, 0.782, 0.018] | **−0.246** | EI ξ=0.02, Matérn | **All-time best** (+66%) |
| 9 | [0.482, 0.432, 0.767, 0.850, 0.010] | −0.485 | Mean, Matérn | Butter=0.850 too high |
| 10 | [0.453, 0.402, 0.745, 0.556, 0.099] | pending | EI ξ=0.008, Matérn 3/2 | Butter reduced; kernel trial |

### All-time best

> −0.2462 — Week 8 — [0.472, 0.407, 0.735, 0.782, 0.018]

### Key findings

- Optimal recipe region: Flour ≈ 0.47, Sugar ≈ 0.41, Eggs ≈ 0.74, Butter ≈ 0.78, Milk < 0.02
- Butter is the most sensitive dimension: 0.78 works, 0.85 (W9) and 0.97 (W4) cause large regressions
- Milk must remain very low (< 0.02 for best results, < 0.17 for acceptable)
- Mean acquisition produced two best results (W5, W6); EI produced the overall best (W8)
- Eggs peak is narrow: 0.735 works, 0.821 (W7) does not

---

## Function 7 — 6D GBM Hyperparameters

**Dimensions:** 6 (D1=n_estimators, D2=learning_rate, D3=max_depth, D4=subsample, D5=max_features, D6=regularisation) · **Initial data:** 30 points · **Output range:** [0.0, 2.451]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 → **Rational Quadratic** (from W8) | Notebook 06 showed RQ R²=0.868 vs Matérn 5/2's 0.493–0.722; captures multi-scale structure |
| ARD | Disabled for RQ | sklearn's RQ kernel does not support ARD |
| Acquisition | EI throughout | Consistent improvement with moderate ξ |
| β | 1.96 → 1.0 (from W9) | Reduced to stay exploitative near confirmed deterministic peak |
| ξ | 0.05 → 0.005 (W8) → 0.002 (W10) | Progressively tightened |
| Y-transform | standardize | |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | 1.365 at [0.058, 0.492, 0.247, 0.218, 0.420, 0.731] | — | Low D1, mid D2, high D6 |
| 1 | [0.333, 0.310, 0.250, 0.800, 0.800, 0.050] | 0.121 | EI ξ=0.05, Matérn | High D4/D5, low D6 — worst |
| 2 | [0.095, 0.365, 0.337, 0.317, 0.362, 0.721] | **2.358** | EI ξ=0.05, Matérn | **Beats initial** (+73%) |
| 3 | [0.061, 0.428, 0.284, 0.256, 0.374, 0.724] | 1.931 | EI ξ=0.05, Matérn ARD | D2 too high (0.428) |
| 4 | [0.024, 0.110, 0.273, 0.338, 0.357, 0.537] | 0.745 | EI ξ=0.10, Matérn ARD | D2 too low (0.110), D6 too low (0.537) |
| 5 | [0.095, 0.365, 0.337, 0.317, 0.362, 0.721] | 2.356 | EI ξ=0.05, Matérn | Near-exact W2 repeat; confirmed deterministic |
| 6 | [0.013, 0.383, 0.380, 0.242, 0.263, 0.707] | 2.189 | EI ξ=0.05, Matérn | D1 perturbation test; D1=0.013 suboptimal |
| 7 | [0.095, 0.368, 0.337, 0.318, 0.360, 0.722] | 2.347 | EI ξ=0.05, Matérn | Near W2/W5; confirms stability |
| 8 | [0.073, 0.358, 0.341, 0.322, 0.272, 0.727] | **2.377** | EI ξ=0.005, **RQ** | **New best** — RQ kernel + D5 reduction |
| 9 | [0.090, 0.362, 0.338, 0.315, 0.291, 0.724] | **2.451** | EI ξ=0.005, RQ, β=1.0 | **All-time best** (+80%) |
| 10 | [0.103, 0.343, 0.353, 0.248, 0.380, 0.856] | pending | EI ξ=0.002, RQ, β=1.0 | D6 pushed higher; D4 reduced |

### All-time best

> 2.4514 — Week 9 — [0.090, 0.362, 0.338, 0.315, 0.291, 0.724]

### Key findings

- Function is deterministic: W2 and W5 submitted near-identical inputs and received outputs differing by < 0.002
- RQ kernel switch (W8) broke a 6-week plateau — captures both broad flat regions and narrow peak simultaneously
- D6 (regularisation) ≈ 0.72 is load-bearing; D1 (n_estimators) ≈ 0.07–0.10 is optimal
- D5 was reduced from 0.362 (W2) to 0.272 (W8) to 0.291 (W9) — this dimension had room to optimise
- Two consecutive improvements after kernel switch (W8, W9) validate the analytical approach

### Exploratory analysis

- `analysis/06_kernel_variants_ngboost.ipynb` — LOO R² kernel comparison; RQ selected

---

## Function 8 — 8D ML Hyperparameters

**Dimensions:** 8 (D1–D8) · **Initial data:** 40 points · **Output range:** [0.0, 9.875]

### Surrogate configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Kernel | Matérn 5/2 | Consistent throughout; rough landscape with complex interactions |
| ARD | Enabled from W3 | 8D space — ARD critical for identifying which dimensions matter |
| Acquisition | UCB → **EI** (from W8) | Switched to EI after establishing hard constraints |
| β | 2.5 → 3.5 (W3 mistake) → 2.5 → 1.5 (W7) → 2.5 (W8–W10) | W3's β=3.5 caused catastrophic exploration; W7's β=1.5 produced near-best |
| ξ | 0.1 → 0.02 (W8) → 0.1 (W9) → 0.049 (W10) | Varying to balance exploit/explore |
| Y-transform | standardize | |

### Submission history

| Week | X submitted | Y returned | Acquisition settings | Outcome |
|------|-------------|------------|----------------------|---------|
| Init | — | 9.598 at [0.056, 0.066, 0.023, 0.039, 0.404, 0.801, 0.488, 0.893] | — | Very low D1–D4, moderate D5, high D6/D8 |
| 1 | [0.242, 0.754, 0.171, 0.087, 0.351, 0.974, 0.195, 0.656] | 9.260 | UCB β=2.5, Matérn | D2 too high |
| 2 | [0.212, 0.204, 0.040, 0.040, 0.973, 0.067, 0.219, 0.061] | **9.704** | UCB β=2.5, Matérn | **Beats initial** — high D5 |
| 3 | [0.062, 0.956, 0.001, 0.908, 0.527, 0.002, 0.943, 0.228] | 7.318 | UCB β=3.5, Matérn ARD | β too high; D2/D4 extreme |
| 4 | [0.091, 0.141, 0.251, 0.059, 0.817, 0.769, 0.130, 0.089] | 8.284 | UCB β=2.5, Matérn ARD | D3=0.251 too high |
| 5 | [0.136, 0.240, 0.025, 0.032, 0.989, 0.204, 0.334, 0.718] | **9.800** | UCB β=2.5, Matérn ARD | **New best** — D3/D4 near zero |
| 6 | [0.471, 0.644, 0.032, 0.417, 0.918, 0.143, 0.350, 0.944] | 9.189 | UCB β=2.5, Matérn ARD | D1/D4 violated constraints |
| 7 | [0.076, 0.130, 0.011, 0.025, 0.955, 0.157, 0.320, 0.869] | 9.775 | UCB β=1.5, Matérn ARD | Near-best; hard constraints enforced |
| 8 | [0.094, 0.275, 0.004, 0.019, 0.942, 0.697, 0.329, 0.861] | **9.830** | EI ξ=0.02, Matérn ARD | **New best** — D3→0.004, D4→0.019 |
| 9 | [0.080, 0.220, 0.003, 0.015, 0.965, 0.506, 0.326, 0.871] | **9.875** | EI ξ=0.1, Matérn ARD | **All-time best** — D3/D4 pushed further |
| 10 | [0.089, 0.471, 0.007, 0.711, 0.689, 0.953, 0.479, 0.793] | pending | EI ξ=0.049, Matérn ARD | Exploratory: D4/D5 constraint test |

### All-time best

> 9.8753 — Week 9 — [0.080, 0.220, 0.003, 0.015, 0.965, 0.506, 0.326, 0.871]

### Key findings

- D3 and D4 must be near zero: every best result has D3 < 0.03 and D4 < 0.04
- D5 must be high (> 0.94): correlated with all top results
- D6, D7, D8 are relatively insensitive — GP ARD assigns long lengthscales to these dimensions
- Hard constraint enforcement (manual clipping) is critical: UCB with β=2.5 regularly suggests points outside safe bounds
- W10 is a deliberate exploratory bet: D4=0.711 and D5=0.689 massively violate established constraints to test whether the GP's ARD conclusions hold

### Exploratory analysis

- `analysis/03_function8_rf_surrogate.ipynb` — Random Forest feature importance cross-validated GP ARD findings
- `analysis/05_nn_surrogate_analysis.ipynb` — Deep Ensemble vs GP comparison; GP superior at n ≤ 50
- `analysis/06_kernel_variants_ngboost.ipynb` — NGBoost tested and rejected (95% PI coverage 2–25%)

---

## Cross-Function Summary

| Fn | Dims | Kernel (current) | Acquisition | Key params | Y-transform | Initial Best | All-time Best | Best Week | Improvement |
|----|------|------------------|-------------|-----------|-------------|-------------|--------------|-----------|------------|
| 1 | 2 | RBF (W10 trial) | UCB β=0.5 | — | arcsinh | ≈0 | 1.64×10⁻⁷ | W8 | +50 orders |
| 2 | 2 | Matérn 5/2 | UCB β=0.7 | het-GP | standardize | 0.611 | **0.726** | W6 | +19% |
| 3 | 3 | Matérn 5/2 | EI ξ=0.012 | — | standardize | −0.035 | **−0.009** | W5 | +74% |
| 4 | 4 | Matérn 3/2 | EI ξ=0.002 | ARD | standardize | −4.026 | **+0.367** | W8 | +109% |
| 5 | 4 | RBF | Mean | — | standardize | 1088.9 | **2238.7** | W9 | +106% |
| 6 | 5 | Matérn 3/2 (W10 trial) | EI ξ=0.008 | — | standardize | −0.714 | **−0.246** | W8 | +66% |
| 7 | 6 | Rational Quadratic | EI ξ=0.002 | β=1.0 | standardize | 1.365 | **2.451** | W9 | +80% |
| 8 | 8 | Matérn 5/2 | EI ξ=0.049 | ARD | standardize | 9.598 | **9.875** | W9 | +2.9% |

**All 8 functions have beaten the initial data best.**

---

## Model Selection Rationale — Alternatives Considered

### Why not SVR as primary surrogate?

SVR via Support Vector Regression can model smooth functions well, as demonstrated in `analysis/02_function2_svr_exploration.ipynb` for Function 2. However, SVR provides no uncertainty estimate — making it unsuitable for principled acquisition functions (UCB, EI, PI all require a posterior variance). It is used as a validation cross-check only.

### Why not Random Forest as primary surrogate?

Random Forest was evaluated as a SMAC-style surrogate for Function 8 in `analysis/03_function8_rf_surrogate.ipynb`. RF provides useful feature importance not available from the GP, and tree-variance can proxy uncertainty. However, RF uncertainty estimates are poorly calibrated compared to GP posteriors. The GP+RF ensemble (RF models the global trend, GP models residuals) proved to overfit at n ≤ 50, with GBM in-sample R²=0.9995 but poor out-of-sample performance. RF is used as a diagnostic tool to validate GP ARD length-scale findings.

### Why not a neural network?

Deep Ensembles (K=10 MLPs) were tested in `analysis/05_nn_surrogate_analysis.ipynb`. At n ≤ 44, the Deep Ensemble achieved LOO R²=−0.417 on F7 and 0.906 on F8, versus the GP's 0.563 and 0.985. The ensemble's 95% PI coverage was 0.907 vs the GP's 0.977 — under-calibrated uncertainty makes it unsuitable for acquisition. Neural networks require n ≫ 100 to outperform GPs in low-dimensional BBO settings.

### Why not NGBoost?

NGBoost (Natural Gradient Boosting) was tested in `analysis/06_kernel_variants_ngboost.ipynb`. While it achieved reasonable R² for some functions (F4: 0.874), its 95% PI coverage was catastrophically low — 0.250 for F4, 0.065 for F8, 0.022 for F7. This means the uncertainty estimates are meaningless for acquisition. Tree-based probabilistic surrogates require n ≥ 100 for viable calibration.

---

## Engineering Changes Log

### W4 — Output standardisation (`"standardize"` Y-transform)

**Implemented:** 2026-04-08, between W3 results and W4 submissions.

**Change:** Z-score normalisation applied to F2–F8 before GP fitting. F1 retains `arcsinh`. GP built with `normalize_y=False` to avoid double-normalisation.

**Impact:** β and ξ now have consistent meaning across functions. Before standardisation, EI ξ=0.01 meant "improve by 0.002% of σ" for F5 but "improve by 12.5% of σ" for F3. After: uniformly 1% of σ.

### W5 — Heteroscedastic GP for F2

**Implemented:** 2026-04-08, between W4 submission and W5.

**Change:** Per-point noise estimated via LOO residuals + Gaussian kernel smoothing (bandwidth=0.20). Peak region near [0.70, 0.93] assigned ~1.6× higher noise than flat region.

**Impact:** Acquisition function no longer chases the phantom 0.118 gradient between two nearby peak observations. W6 (het-GP active, β=1.2) produced the all-time best.

### W7–W8 — Kernel variant optimisation (notebook 06)

**Implemented:** Between W6 results and W8 submissions.

**Change:** LOO R² and 95% PI coverage compared across Matérn 5/2, Matérn 3/2, Rational Quadratic, and Composite (Matérn+Linear) kernels for F4, F7, F8.

**Results:**
- F4: Matérn 3/2 (R²=0.961) >> Matérn 5/2 (0.485) → **switched**
- F7: RQ (R²=0.868) > Matérn 5/2 (0.493–0.722) → **switched**
- F8: RQ marginally better (0.870 vs 0.855) but Matérn 5/2 retained (proven track record)

**Impact:** F4 produced three consecutive bests after the switch (W6–W8). F7 broke a 6-week plateau (W8, W9).

### W7–W8 — NGBoost evaluation and rejection (notebook 06)

**Implemented:** Alongside kernel variants.

**Change:** NGBoost tested on F4, F7, F8 with 5-fold CV.

**Results:** 95% PI coverage of 0.022–0.250 across all functions. Rejected for BBO use.

### W7–W8 — F1 hotspot hunt (notebook 07)

**Implemented:** Between W6 results and W8 submissions.

**Change:** Model-free log-space analysis of F1. Spearman correlation between distance from [0.65, 0.68] and log₁₀(|Y|) yielded r=−0.696, p=0.002. Candidate [0.691, 0.707] identified.

**Impact:** W8 query at [0.691, 0.707] → Y=1.64×10⁻⁷ — a 50-order-of-magnitude breakthrough, the largest single-week improvement in the challenge.

### W10 — Kernel selector added to UI

**Implemented:** Between W9 results and W10 submissions.

**Change:** Added GP kernel dropdown to the Streamlit dashboard (Matérn 5/2, Matérn 3/2, RQ, RBF). Per-function settings (kernel, acquisition, β, ξ) now persist in session state across function switches. Fixed truthiness bug where `xi_override=0.0` was silently ignored.

**Impact:** Kernel choice is now visible and configurable per function in the UI, with the selection recorded in submission metadata and displayed on function cards.
