# Strategy Log

High-level decisions and lessons that span multiple functions or weeks. Updated when strategy evolves.

---

## Overall Approach

**Surrogate model:** Gaussian Process with Matérn 5/2 kernel (default) or RBF where the function is known/believed to be smooth.

**Acquisition functions in use:**

- **UCB** (Upper Confidence Bound): `μ + β·σ` — favours uncertain regions. Used for noisy/unexplored functions (F1, F2, F4, F8).
- **EI** (Expected Improvement): Expected gain over current best. Used when a clear incumbent exists (F3, F5, F6, F7).

**Training data:** Initial `.npy` observations (challenge-provided) combined with portal submissions. The GP is refitted from scratch each time a new observation is added.

---

## Week 1 → Week 2 Strategic Shift

### Key finding

Week 1 submissions were generated **before** the initial `.npy` data was integrated into the surrogate model. The GP was trained on near-zero data, producing suggestions that largely ignored the strong signals in the initial data. All 8 functions returned results below their initial best.

### Actions taken

- Initial data integration implemented: GP now trains on 10–40 initial points + portal submissions
- Function 1 kernel changed from RBF to Matérn 5/2
- Per-observation metadata tracking added (acq function, β, ξ, kernel)

### Direction for week 2

**Across all functions:** Reduce exploration pressure — the initial data already reveals where the promising regions are. Most functions should shift toward EI with low ξ (0.01–0.05) to exploit known good regions.


| Fn  | Direction    | Key move                                                   |
| --- | ------------ | ---------------------------------------------------------- |
| F1  | Explore      | Systematic quadrant coverage — hotspot not found yet       |
| F2  | Exploit      | Move toward [0.75, 0.93] — confirmed X₁ > 0.7 region       |
| F3  | Exploit      | Stay near [0.49, 0.61, 0.34] — confirmed B≈0.6 matters     |
| F4  | Exploit      | Target [0.55, 0.38, 0.50, 0.18] — confirmed P4 low is good |
| F5  | Exploit hard | Target [0.18, 0.87, 0.90, 0.90] — 21× gap to close         |
| F6  | Exploit      | Reduce Sugar and Milk further — confirmed penalising       |
| F7  | Exploit      | Perturb [0.058, 0.49, 0.25, 0.22, 0.42, 0.73]              |
| F8  | Exploit      | Push P1-P4 lower, keep P6/P8 high                          |


---

## Kernel Selection Rationale


| Kernel         | Functions                  | Reason                                                                                                                                             |
| -------------- | -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Matérn 5/2** | F1, F2, F3, F4, F6, F7, F8 | Default for unknown/rough/noisy functions. Twice differentiable — realistic assumption for most real-world processes.                              |
| **RBF**        | F5                         | Function described as unimodal with a clean peak — consistent with infinite differentiability. Only appropriate when smoothness is well-supported. |


**Rule:** Do not change kernel week-to-week. Kernel encodes a structural assumption about the function — it should be a deliberate one-time choice, not a tuning parameter.

---

## Exploration vs Exploitation by Phase

### Phase 1 (Weeks 1–2): Establish landscape

- Use higher β/ξ to cover the space
- Goal: identify which regions are promising, which are not
- Expected: many queries below initial best

### Phase 2 (Weeks 3–5): Focused exploitation

- Reduce β (UCB) to 1.0–1.5 for well-understood functions
- Reduce ξ (EI/PI) to 0.01 for most functions
- Submit within the confirmed best neighbourhood

### Phase 3 (Weeks 6+): Final convergence

- Near-greedy exploitation: EI ξ ≈ 0.001 or pure GP mean maximisation
- One last exploratory query only if confidence in current best is low

---

## Week 3 → Week 4 Strategic Shift

### What happened in week 4

Week 4 produced mixed results: one new all-time best (F2), five regressions (F3, F4, F5, F6, F7), one partial recovery (F8), and one continued zero (F1).

The dominant failure mode was **EI/UCB over-exploration in late-stage search.** With ξ=0.05–0.1, the acquisition function repeatedly identified high-uncertainty unexplored regions as more promising than the known best neighbourhood. At n=20–40 points in 5–8 dimensions, the GP always has substantial uncertainty in remote regions, so high-ξ EI will always be pulled toward them.

### Key finding: ξ must drop sharply after week 3


| Phase     | Appropriate ξ | Reason                                                |
| --------- | ------------- | ----------------------------------------------------- |
| Weeks 1–2 | 0.05–0.10     | Need to explore; current best is provisional          |
| Week 3    | 0.02–0.05     | Converging on best region; reduce jumps               |
| Weeks 4+  | 0.005–0.02    | Best neighbourhood confirmed; tight exploitation only |


The F7 regression is the clearest illustration: two consecutive weeks of ξ=0.05 and ξ=0.1 after finding the best at W2 produced queries 18% and 68% below the known peak.

### Surrogate model improvements made before week 4

Three changes were deployed to `capstone_app.py` between W3 and W4 — see REFLECTIONS.md for full technical detail:

1. **Y-standardisation (all functions except F1):** Z-scores applied before GP fitting. β and ξ now have consistent cross-function meaning. Primary beneficiaries: F4 (large outlier at −21.254 distorted kernel fit) and F5 (27× Y range now compressed to unit variance). Full analysis in REFLECTIONS.md § "Surrogate Model Improvement — Output Standardisation."
2. **Heteroscedastic GP for F2:** Leave-one-out residuals used to assign per-point noise variance, with Gaussian kernel smoothing. The peak region at [0.70, 0.93] had contradictory observations (0.611 vs 0.493); the het-GP correctly assigned these higher noise rather than treating the gap as a genuine gradient. F2 achieved a new all-time best in W4 (0.648 vs 0.611 previous best) — direct validation of this change. Full analysis in REFLECTIONS.md § "Surrogate Model Improvement — Heteroscedastic GP."
3. **GP vs GBM vs GP+GBM ensemble analysis for F8:** Tested in `analysis/04_function8_gpgbm_ensemble.ipynb`. The GP (ARD Matérn) achieved CV R²=0.969 vs GBM's LOO R²=0.798. The ensemble failed because GBM overfitted the small dataset (in-sample R²=0.9995 with n=43), leaving residuals near zero and collapsing the residual GP to constant posterior. Key finding: despite different predictive accuracy, both GP and GBM agree that D3 and D1 are the dominant dimensions for F8 (identical ranking). Decision: continue with GP as sole surrogate. Full analysis in REFLECTIONS.md § "Surrogate Comparison — GP vs GBM vs GP+GBM Ensemble."

### Neural network viability analysis

Tested in `analysis/05_nn_surrogate_analysis.ipynb`. Summary of findings:

- **CNNs:** Not applicable to any function. CNN's inductive bias (local spatial correlation in grid inputs) doesn't exist in hyperparameter/ingredient/compound spaces where dimension order has no spatial meaning. For F1/F2 (genuine 2D spatial inputs) the GP's Matérn kernel is provably more data-efficient at n=14.
- **MLPs:** Not viable at current sample sizes. Even the smallest MLP (16→8 units, 193 parameters) has 7–14× more free parameters than training points for every function. LOO R² for F7 (n=33, 6D): GP=0.563, MLP=−0.091, Deep Ensemble=−0.417.
- **When would NNs become viable?** Approximately n≥100–200 observations in 8D. Most promising future direction: Deep Kernel Learning (NN feature extractor feeding a GP), viable if any function accumulates ~60+ observations.

Decision: GP remains the production surrogate for all functions. Full analysis in REFLECTIONS.md § "Neural Networks and CNNs as BBO Surrogates."

### Direction for week 5


| Fn  | All-time best      | W4 result | W4 failure mode              | W5 action                                   |
| --- | ------------------ | --------- | ---------------------------- | ------------------------------------------- |
| F1  | ≈0 (no signal)     | ≈0        | Revisited tested region      | Lower-left probe [0.08, 0.48]               |
| F2  | **0.648 (W4 NEW)** | 0.648 ↑   | —                            | Tight continuation near [0.700, 0.961]      |
| F3  | −0.018 (W2)        | −0.064    | High A=0.983                 | Return to [0.439, 0.461, 0.503], ξ=0.02     |
| F4  | −1.177 (W2)        | −2.370    | P3=0.499 too high            | Lock P3≈0.270, return to W2-like coords     |
| F5  | 1374.52 (W3)       | 1124.92   | C1/C4 drift with "mean" acq  | EI ξ=0.01, return C4→0.872                  |
| F6  | −0.384 (W3)        | −1.294    | Butter=0.969 catastrophic    | "Mean" acq, return to W3 coords             |
| F7  | 2.358 (W2)         | 0.745     | ξ=0.1 large jump, low lr/reg | Submit W2 best exactly: [0.095, 0.365, ...] |
| F8  | 9.704 (W2)         | 8.284     | D3=0.251, D6=0.769 too high  | D3<0.07, D6<0.15 hard constraints           |


---

## Week 5 → Week 6 Strategic Shift

### What happened in week 6

Week 6 produced two new all-time bests (F2: 0.726, F4: +0.136, F6: −0.296), two confirmed-safe regressions (F3: −0.013, F7: 2.189), and two acquisition-driven failures (F5: 1223, F8: 9.189). F1 continued to return effectively zero.

The dominant pattern in the failures was **UCB over-exploration when hard constraints were not enforced**. F8's acquisition with β=2.5 produced a query at D1=0.471, D4=0.417 — both far outside confirmed safe bounds. F5's D4 reduction from 0.872 to 0.811 violated the D4 > 0.87 hard constraint. Both failures were preventable.

The dominant pattern in the successes was **pure exploitation with GP posterior mean** (F6) and **het-GP tight targeting** (F2). The "mean" acquisition has now produced three consecutive improvements for F6 — the most consistent run of any function. F4's breakthrough to positive Y came from UCB remaining close to the W5 coordinates.

### Key finding: manual constraint enforcement is now mandatory

At this stage of the search (6–7 queries submitted, n = 30–50 including initial data), the GP has enough uncertainty in remote regions to override all prior knowledge if β or ξ is not tightly managed. The solution is to **clip any suggested query outside confirmed bounds before submitting**, regardless of what the acquisition function recommends.


| Acquisition | Appropriate β/ξ | Notes                                             |
| ----------- | --------------- | ------------------------------------------------- |
| UCB         | β ≤ 1.5         | Reduce from 2.5 — only safe with hard clipping    |
| EI          | ξ ≤ 0.01        | "Mean" (ξ≈0) is preferred for converged functions |
| Mean        | N/A             | Default for F6; consider for F3, F5, F7           |


### Direction for week 7


| Fn  | All-time best       | W6 result | W6 outcome           | W7 action                                                                          |
| --- | ------------------- | --------- | -------------------- | ---------------------------------------------------------------------------------- |
| F1  | ≈0                  | ≈0        | Cluster exhausted    | Abandon cluster — probe [0.08, 0.20] (lower-left)                                  |
| F2  | **0.726 (W6 NEW)**  | 0.726 ↑   | New best             | [0.699, 0.920] — probe X₂ slightly lower                                           |
| F3  | −0.009 (W5)         | −0.013    | Stable near-best     | [0.438, 0.462, 0.505] — micro-nudge, ξ=0.005                                       |
| F4  | **+0.136 (W6 NEW)** | +0.136 ↑  | First positive ever  | [0.450, 0.420, 0.365, 0.378] — stay very close, EI ξ=0.01                          |
| F5  | 1412.6 (W5)         | 1223      | D4 regression        | Return to W5 exact: [0.339, 0.838, 0.946, 0.872]                                   |
| F6  | **−0.296 (W6 NEW)** | −0.296 ↑  | New best             | [0.410, 0.415, 0.780, 0.785, 0.015] — push Eggs↑, Milk↓                            |
| F7  | 2.357 (W2/W5)       | 2.189     | D1 test complete     | Return to W2/W5 exact: [0.095, 0.365, 0.337, 0.317, 0.362, 0.721]                  |
| F8  | 9.800 (W5)          | 9.189     | Constraint violation | [0.130, 0.235, 0.025, 0.030, 0.985, 0.200, 0.330, 0.720] — enforce all hard limits |


---

## Week 7 → Week 8 Strategic Shift

### What happened in week 8

**Best week of the entire challenge: 6 new all-time bests** (F1, F4, F5, F6, F7, F8). Only F2 and F3 did not set new bests.

The dominant pattern in the successes: **engineering-driven improvement, not just GP exploitation.** Four of the six bests were directly enabled by between-weeks analysis:

- F1: model-free hotspot hunt (notebook 07) identified [0.691, 0.707] → Y jumped by 50 orders of magnitude
- F4: Matérn 3/2 kernel switch (notebook 06) → third consecutive improvement
- F7: Rational Quadratic kernel (notebook 06) + tight ξ → first improvement over W2 after six weeks
- F5: releasing the D2 constraint from [0.83, 0.85] to 0.91 → 32% yield jump

F8's improvement came from reduced β (1.5 vs 2.5) enforcing hard dimension constraints, and F6 recovered from the W7 regression via GP mean acquisition.

### Direction for week 9


| Fn  | All-time best         | W8 result  | W8 outcome            | W9 action                                                  |
| --- | --------------------- | ---------- | --------------------- | ---------------------------------------------------------- |
| F1  | **1.6×10⁻⁷ (W8 NEW)** | 1.6×10⁻⁷ ↑ | Breakthrough          | [0.670, 0.695] — toward magnitude centre                   |
| F2  | 0.726 (W6)            | 0.715      | Near-best             | [0.699, 0.932] — stay tight                                |
| F3  | −0.009 (W5)           | −0.017     | Drifting              | Return to W5 exact: [0.439, 0.461, 0.503]                  |
| F4  | **+0.367 (W8 NEW)**   | +0.367 ↑   | 3rd consecutive       | [0.438, 0.431, 0.355, 0.380] — stay tight                  |
| F5  | **1963.7 (W8 NEW)**   | 1963.7 ↑   | Massive jump          | [0.350, 0.923, 0.961, 0.880] — push D2 higher              |
| F6  | **−0.246 (W8 NEW)**   | −0.246 ↑   | Recovered             | [0.475, 0.410, 0.740, 0.785, 0.015] — stay tight           |
| F7  | **2.377 (W8 NEW)**    | 2.377 ↑    | 6-week plateau broken | [0.073, 0.358, 0.341, 0.322, 0.260, 0.727] — test D5 lower |
| F8  | **9.830 (W8 NEW)**    | 9.830 ↑    | D3/D4→0 confirmed     | [0.080, 0.220, 0.003, 0.015, 0.965, 0.500, 0.326, 0.871]   |


---

## Week 8 → Week 9 Strategic Shift

### What happened in week 9

**3 new all-time bests** (F5: 2238.7, F7: 2.451, F8: 9.875) and 5 regressions. The theme: tight exploitation along confirmed gradients works; speculative jumps do not.

F5 improved for the fourth consecutive week by pushing D2/D3/D4 higher — each increment smaller than the last. F7 and F8 improved via micro-perturbation (move distances < 0.10) within confirmed basins. Meanwhile F4's speculative probe at D1=0.230 — deliberately violating the [0.35, 0.45] hard constraint — returned −3.21, the worst result since W1.

### Direction for week 10


| Fn  | All-time best       | W9 result  | W9 outcome          | W10 action                                                                          |
| --- | ------------------- | ---------- | ------------------- | ----------------------------------------------------------------------------------- |
| F1  | 1.6×10⁻⁷ (W8)       | −1.3×10⁻¹⁴ | Regression          | Switch to RBF kernel, β=0.5 — test infinite smoothness for steep hotspot            |
| F2  | 0.726 (W6)          | 0.559      | 3rd regression      | Drop β from 2.5→0.7 — near-pure exploitation                                        |
| F3  | −0.009 (W5)         | −0.014     | 4th plateau week    | Radical corner probe [0.99, 0.97, 0.95] — test for second basin                     |
| F4  | +0.367 (W8)         | −3.213     | Speculative failure | Return to validated region, ξ→0.002 — damage recovery                               |
| F5  | **2238.7 (W9 NEW)** | 2238.7 ↑   | 4th consecutive     | Continue gradient: D2→0.95, D3→0.985, D4→0.98                                       |
| F6  | −0.246 (W8)         | −0.427     | Regression          | Switch to Matérn 3/2 + EI, Butter→0.556 — exploratory bet                           |
| F7  | **2.451 (W9 NEW)**  | 2.451 ↑    | New best            | EI ξ=0.002, push D6→0.856 — test if D6 extends further                              |
| F8  | **9.875 (W9 NEW)**  | 9.875 ↑    | New best            | High-risk constraint violation: D4→0.71, D5→0.69 — test if ARD is over-constraining |


---

## Week 9 → Week 10 Strategic Shift

### What happened in week 10

**1 new all-time best** — F5 (3448.2, +54%, sixth consecutive improvement). All 7 other functions regressed or stagnated.

The dominant pattern: **exploitation along a confirmed monotonic gradient is the only consistently successful strategy at this sample budget.** F5's RBF/mean combination has produced improvements every week from W5 to W10. Every deliberate exploratory bet in W10 failed:

- F3's corner probe (Y = −0.237 vs best −0.009) — no second basin exists
- F6's Butter reduction (Y = −0.389 vs best −0.246) — hard constraint validated
- F8's massive constraint violation (Y = 9.166 vs best 9.875) — D4 and D5 constraints validated

### End-of-challenge summary of strategy effectiveness


| Strategy                                                      | Win rate   | Notable examples                                |
| ------------------------------------------------------------- | ---------- | ----------------------------------------------- |
| Tight exploitation (move < 0.05, same kernel/acq)             | ~65%       | F5 W5–W10 (6/6), F4 W6–W8 (3/3), F7 W9          |
| Kernel switch within confirmed basin                          | ~50%       | F4 Matérn 3/2 (success), F1 RBF (failure)       |
| Speculative exploration (move > 0.30 or constraint violation) | ~10%       | F8 W10 (fail), F4 W9 (fail), F3 W10 (fail)      |
| Model-free analysis → targeted query                          | 100% (n=1) | F1 W8 hotspot hunt                              |
| Between-weeks surrogate engineering                           | ~80%       | Kernel variants, NGBoost evaluation, F1 hotspot |


---

## Week 10 → Week 11 Strategic Shift

### What happened between W10 and W11

Deep landscape analysis revealed the **root cause of most W6–W10 regressions**: the GP acquisition function was suggesting queries in known-bad regions because candidates were sampled uniformly from [0,1]^d. The uncertainty term (σ·φ(z) in EI, β·σ in UCB) made high-uncertainty corners competitive with genuine improvement near the basin.

Three structural engineering changes were implemented:

1. **Per-function search bounds** (F3, F4, F5, F7, F8) — candidates now sampled from confirmed basins. This is the most impactful single change since output standardisation in W4.
2. **Dimension masking** (F8) — GP fitted on 6 of 8 dimensions. D6 and D8 are noise (top-5 spread > 0.6, Spearman |ρ| < 0.02). Reducing from 8D to 6D improves the data-to-dimension ratio from 6.25 to 8.33.
3. **Length-scale cap** (F3, F8) — upper bound reduced from 10.0 to 3.0. F3's D1 previously learned LS=4.18, making the GP see a flat function.

### W11 strategic posture by function

| Fn | Strategy | Rationale |
|----|----------|-----------|
| F1 | Return to Matérn 5/2, tight near W8 hotspot | W10 RBF trial failed (3 orders worse). Matérn's finite differentiability better matches steep radial decay |
| F2 | Reduce β from 0.7 to 0.5, stay near peak | Noise-dominated; any query >0.01 from [0.70, 0.93] is wasted |
| F3 | First bounded query, EI ξ=0.005 | Basin-constrained for the first time; GP now forced to learn local structure |
| F4 | Bounded, ξ reduced to 0.001 | Tightest exploitation yet; all 4 dims within confirmed [0.39–0.51] |
| F5 | Continue gradient push, D2–D4 → 1.0 | 6 consecutive bests; no reason to change strategy |
| F6 | Return to Matérn 5/2, reduced β and ξ | W10's Matérn 3/2 and exploration failed; revert to reliable config |
| F7 | Bounded, RQ kernel, ξ=0.01 | Search bounds prevent the known D4/D5 drift |
| F8 | 6D GP, bounded, UCB β=0.4 | Most heavily engineered function — all three changes applied simultaneously |

---

## Week 11 → Week 12 Strategic Shift

### W11 results — search bounds validated

**3 new all-time bests** — F3 (−0.00831, first improvement since W5), F5 (4125.9, 7th consecutive), F7 (2.588, +5.6%). The search bounds directly enabled F3 and F7; F5 continued its gradient push.

### The confidence shift

W12 marks a strategic pivot from surrogate-driven to analysis-driven query selection. Two new analysis notebooks (10: surrogate alternatives, 11: PCA) showed that:

- The GP is actively misleading for 3 functions (F1, F2, F3)
- Model-free weighted centroids outperform the GP for stale functions (F6 centroid dist=0.029 vs GP dist=0.131)
- Local GP (TuRBO-style, k=15 nearest) dramatically outperforms global GP for F3 (LOO R² 0.511 vs −0.014)
- NN-64×2 does NOT improve over GP for F6 (R² delta = +0.003, not significant)
- PCA confirms all dimensions are needed (no input-space reduction), but is uncorrelated with output-relevance measures

### W12 strategic posture

| Fn | Strategy | Source |
|----|----------|--------|
| F1 | Model-free radial, β=0.1 | Micro-perturbation toward hotspot centre |
| F2 | AI analysis + centroid, β=0.3 | Noise-dominated; no surrogate helps |
| F3 | Local GP + centroid blend | Notebook 10's local GP argmax at dist=0.094 |
| F4 | Top-5 centroid (surrogate ignored) | GP R²=0.94 but W11 still regressed; centroid is safer |
| F5 | Gradient push (D2/D3/D4 → 1.0) | 7th consecutive best; near boundary |
| F6 | **Model-free centroid directly** | Notebook 10: centroid closer than GP or NN |
| F7 | Bounded GP + centroid check | Near W11 best; search bounds active |
| F8 | AI analysis + manual, EI β=2.5 | Hand-tuned toward W9 best; tight dims constrained |

---

## Hard Dimension Constraints (Evidence-Based)

These constraints are derived from multiple weeks of observation and should not be violated without strong analytical justification:


| Fn  | Constraint                                                                     | Evidence                                                                                             | W11 status                                                  |
| --- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| F1  | X₁ ∈ [0.65, 0.72]; X₂ ∈ [0.68, 0.73]                                           | W8 at [0.691, 0.707] → 1.6×10⁻⁷; W10 at [0.702, 0.721] → 3.7×10⁻¹⁰ (still in band but weaker)        | Confirmed — W11 at [0.700, 0.715], within band              |
| F2  | X₁ ∈ [0.69, 0.71]; X₂ ∈ [0.92, 0.94]                                           | W6 best at X₂=0.932; W10 at [0.706, 0.934] → 0.564 (noise-dominated)                                 | Confirmed — W11 at [0.700, 0.935], within band              |
| F3  | A ∈ [0.35, 0.60]; B ∈ [0.36, 0.56]; C ∈ [0.43, 0.55]                           | W5 best at [0.439, 0.461, 0.503]; W10 corner probe → −0.237. Search bounds now enforced               | **Enforced** — search bounds active; W11 within basin       |
| F4  | D1 ∈ [0.39, 0.51]; D2 ∈ [0.36, 0.54]; D3 ∈ [0.26, 0.42]; D4 ∈ [0.27, 0.46]   | W6–W8 three consecutive bests; W9 D1=0.23 → −3.21; W10 P2=0.484 → −1.43. Search bounds now enforced  | **Enforced** — search bounds active; W11 within basin       |
| F5  | C1 ∈ [0.28, 0.40]; C2 → 1.0; C3 → 1.0; C4 → 1.0                                | W10 best at [0.332, 0.951, 0.985, 0.980] → 3448.2. Search bounds enforce gradient corridor            | **Enforced** — D2–D4 upper bounds set to 1.0               |
| F6  | Flour ∈ [0.40, 0.50]; Eggs ∈ [0.73, 0.77]; Butter ∈ [0.77, 0.83]; Milk < 0.02  | W8 best at Butter=0.782; W10 Butter=0.556 → −0.389 (regression)                                      | Confirmed — no search bounds (F6 stays within naturally)    |
| F7  | D1 ∈ [0.02, 0.15]; D2 ∈ [0.31, 0.42]; D3 ∈ [0.29, 0.39]; D4 ∈ [0.27, 0.37]; D5 ∈ [0.22, 0.41]; D6 ∈ [0.67, 0.78] | W9 best; W10 D6=0.856 → −26%. Search bounds now enforced                               | **Enforced** — all 6 dims bounded; W11 within basin         |
| F8  | D1 < 0.22; D3 < 0.05; D4 < 0.05; D5 > 0.93; D7 ∈ [0.19, 0.36]                 | W9 best; W10 massive violation → −7.2%. Search bounds + 6D GP + LS cap now enforced                   | **Enforced** — search bounds + dim mask + LS cap all active |


---

## Lessons Learned


| Week | Lesson                                                                                                                                                                                                                                                                       |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1    | **Always integrate all available data before generating suggestions.** The initial .npy data is the most important training signal; ignoring it wastes queries.                                                                                                              |
| 1    | **Meta-tracking matters.** Without recording acq/β/ξ/kernel per submission, it is impossible to review why a particular query was made. Now tracked automatically.                                                                                                           |
| 1    | **Kernel choice is a one-time structural decision.** RBF was wrong for F1 — changed to Matérn after analysis. Not appropriate to oscillate between kernels.                                                                                                                  |
| 1    | **Over-exploration in week 1 has downstream cost.** With ~8 remaining queries per function, any query below the initial best is a wasted opportunity.                                                                                                                        |
| 3–4  | **High ξ (0.05–0.1) is explorative even under EI label.** EI with ξ=0.1 behaves more like UCB than like exploitation in 5–8D with ~30 observations. Reduce ξ aggressively after the best region is confirmed.                                                                |
| 4    | **Standardise Y before fitting.** Without standardisation, β and ξ have different real-world meanings for each function. The arcsinh and z-score transforms were the most impactful surrogate improvements made.                                                             |
| 4    | **Heteroscedastic GP for explicitly noisy functions.** F2's per-point noise model directly contributed to the new all-time best in W4. Standard GP with constant alpha misread noise as spatial gradient.                                                                    |
| 4    | **GBM cannot replace GP at current dataset sizes.** GBM achieves LOO R²=0.798 vs GP's 0.969 for F8, and memorises training data (in-sample R²=0.9995). Cross-validation is essential for surrogate model selection — in-sample metrics are meaningless in the BBO context.   |
| 4    | **GP+GBM ensemble breaks when GBM overfits.** Residual GP receives near-zero residuals, collapsing its posterior std ≈ 0 → acquisition function explores nothing. Always validate ensemble components individually.                                                          |
| 4    | **Deep Ensemble (K=10) underestimates uncertainty.** 95% PI coverage = 0.907 vs target 0.950 for F8 — all K models agree in extrapolation. GP's coverage = 0.977. Use GP for calibrated uncertainty.                                                                         |
| 6    | **Enforce hard dimension constraints manually.** UCB with β=2.5 will override any implicit boundary knowledge and push queries into known-bad regions. If the acquisition suggests a value outside a confirmed constraint, clip it before submitting.                        |
| 6    | **Single-dimension perturbation tests have clear ROI.** F7's D1 test (0.095 → 0.013) returned a crisp answer in one query. When the optimum is narrow and deterministic, this protocol is more informative than general EI exploration.                                      |
| 6    | **Cluster exhaustion is a valid stopping criterion.** Six F1 queries in the same region with monotonically decreasing Y confirms the hotspot is elsewhere. Move to a completely new sub-domain rather than continuing to refine within a barren cluster.                     |
| 6    | **Positive Y for F4 confirms that BBO rediscovers ML tuning intuition.** The "moderate everything" optimum at [0.45, 0.42, 0.36, 0.38] is consistent with regularisation theory — extreme settings on any dimension produce worse models.                                    |
| 6–7  | **Kernel choice is NOT a one-time decision.** F4's Matérn 5/2 achieved LOO R²=0.485 — barely better than a mean predictor. Switching to Matérn 3/2 improved this to 0.961. Kernel selection must be re-evaluated as data accumulates and the landscape is better understood. |
| 6–7  | **Rational Quadratic captures multi-scale structure.** For F7, RQ (0.868) outperformed Matérn 5/2 (0.493–0.722) by modelling both the broad flat landscape and the narrow peak simultaneously. Useful when a function has features at different length-scales.               |
| 6–7  | **NGBoost 95% PI coverage is dangerously under-calibrated at small n.** Coverage of 2–25% means the model's uncertainty is meaningless for acquisition. Tree-based probabilistic surrogates require n ≥ 100 to be viable.                                                    |
| 6–7  | **When the GP fails, go model-free.** F1's 183-order dynamic range defeats any kernel-based model. Spearman rank correlation on distance (r=−0.696, p=0.002) revealed the hotspot structure that the GP could not detect.                                                    |
| 6–7  | **Initial data is a designed experiment — read it first.** The challenge designers placed F1's two highest-magnitude points near [0.65–0.73, 0.68–0.73]. Recognising this as a deliberate bracket of the hotspot should have been the week 1 strategy, not the week 7 one.   |
| 8    | **Between-weeks analysis is the highest-ROI activity.** 4 of 6 new bests in W8 were directly enabled by between-weeks engineering (kernel variants, F1 hotspot hunt). Investing time in surrogate diagnostics pays off more than individual query tuning.                    |
| 8    | **Constraints that were "confirmed" can still be wrong.** F5's D2 was held at [0.83, 0.85] for four weeks based on early data. Releasing it to 0.915 produced a 32% yield jump — the constraint was based on too few observations in the high-D2 region.                     |
| 8    | **F1 validated: model-free beats model-based when dynamics are extreme.** The radial Spearman analysis predicted the hotspot correctly from 17 points where the GP could not. For extreme-dynamic-range functions, spatial statistics outperform kernel methods.             |
| 9    | **Speculative probes that violate hard constraints are almost never worth it.** F4's D1=0.230 (outside [0.35, 0.45]) returned the worst score in 8 weeks. The expected information gain from a single far-out probe is too low to justify the query cost at n ≤ 50.          |
| 9    | **Consecutive tight exploitation is the most reliable strategy.** F5 improved 4 consecutive weeks (W5–W9) with move distances < 0.08. F7 and F8 set new bests via micro-perturbation. The pattern is clear: once a basin is confirmed, stay close.                           |
| 10   | **F5 proves monotonic exploitation can work all 10 weeks.** Six consecutive bests (W5→W10) by pushing D2/D3/D4 toward 1.0 — the only function that improved on the final query. RBF/mean was the most consistent surrogate-strategy pairing in the entire challenge.         |
| 10   | **RBF is not universally better for smooth functions.** F1's W10 RBF kernel trial returned 3.7×10⁻¹⁰ vs Matérn's 1.6×10⁻⁷ — three orders of magnitude worse. The steep radial decay favours Matérn's finite differentiability.                                               |
| 10   | **Exploration fails reliably in the final weeks.** F3 corner probe (−0.237), F6 Butter reduction (−0.389), F8 constraint violation (9.166) — all three deliberate exploratory bets regressed. By W10, exploitation is the only rational strategy.                            |
| 10   | **Noise-dominated functions cannot be improved by query tuning alone.** F2's last 5 submissions (W6–W10) all targeted X within 0.01 of the best, yet only W6 found the peak. The limiting factor is irreducible stochasticity, not surrogate quality.                        |
| 10   | **Hard constraint tables should be a W1 deliverable, not a W6 one.** Retrospectively, building and enforcing the constraint table from week 1 would have saved at least 8–10 wasted queries across all functions.                                                            |
| 10–11 | **The candidate generation region is the most impactful hyperparameter.** Sampling 5000 candidates from [0,1]^d means most fall far from the basin. The acquisition's σ·φ(z) term makes high-uncertainty corners competitive, overriding prior knowledge. Search bounds eliminate this structurally. |
| 10–11 | **Dimension masking outperforms ARD in data-sparse regimes.** F8's 8D ARD gave D5 a length-scale of 10.0 (upper bound). Dropping D6/D8 and fitting a 6D GP produced length-scales under 1.0. When n/d < 8, manually identifying noise dims beats ARD. |
| 10–11 | **Length-scale upper bounds should be function-specific.** Default LS bound of 10.0 lets the optimizer learn length-scales 10× the domain width. Capping at 3.0 for F3/F8 forces tighter structure. |
| 10–11 | **GP LOO R² < 0.20 means the surrogate is useless.** F3's R² was 0.17 for all kernels. Radial Spearman (ρ=−0.718) provided more signal from 25 points than any GP could. |
| 10–11 | **5 of 8 acquisition functions wander when unconstrained.** Only F1, F2, F6 stay within the top-5 bounding box. This was the hidden cause of many W6–W10 regressions. |
| 11   | **Search bounds produce immediate results.** F3 had been stale for 6 weeks; the first bounded query set a new best. F7 broke a 2-week plateau. 3 of 8 functions improved simultaneously — the highest hit rate since W8. |
| 11   | **F5's gradient is the most reliable signal in the challenge.** 7 consecutive improvements (W5–W11) by pushing D2/D3/D4 toward 1.0. Total improvement from initial best: +279%. No other strategy has this consistency. |
| 11–12 | **When surrogate confidence is low, prefer model-free candidates.** The top-5 weighted centroid is within 0.04 of the best for all 8 functions. For F6, the centroid (dist=0.029) beat the GP (0.131) and NN (0.175) as a candidate generator. |
| 11–12 | **Local GP (TuRBO-style) solves the global length-scale pathology.** F3's global GP has R²=−0.014 and predicts the optimum at the opposite corner. Local GP (k=15) reaches R²=0.511 with argmax dist=0.094. |
| 11–12 | **NN surrogates do not help at n=31.** NN-64×2 for F6 showed R²=0.693±0.019 vs GP 0.690 — statistically identical. Single-seed results are misleading; always test across multiple seeds. |
| 11–12 | **PCA is the wrong tool for BBO dimension selection.** PCA importance is uncorrelated with RF/Spearman for F8 (ρ=0.071). PCA measures input spread, not output relevance. Use supervised methods (GP ARD, RF importance, Spearman). |
| 12   | **Analysis notebooks have become the primary decision tool.** F6's W12 query is the raw centroid from notebook 10. F4's is the top-5 centroid. The Streamlit dashboard is now a cross-check, not the primary engine. |

