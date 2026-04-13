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

| Fn | Direction | Key move |
|----|-----------|----------|
| F1 | Explore | Systematic quadrant coverage — hotspot not found yet |
| F2 | Exploit | Move toward [0.75, 0.93] — confirmed X₁ > 0.7 region |
| F3 | Exploit | Stay near [0.49, 0.61, 0.34] — confirmed B≈0.6 matters |
| F4 | Exploit | Target [0.55, 0.38, 0.50, 0.18] — confirmed P4 low is good |
| F5 | Exploit hard | Target [0.18, 0.87, 0.90, 0.90] — 21× gap to close |
| F6 | Exploit | Reduce Sugar and Milk further — confirmed penalising |
| F7 | Exploit | Perturb [0.058, 0.49, 0.25, 0.22, 0.42, 0.73] |
| F8 | Exploit | Push P1-P4 lower, keep P6/P8 high |

---

## Kernel Selection Rationale

| Kernel | Functions | Reason |
|--------|-----------|--------|
| **Matérn 5/2** | F1, F2, F3, F4, F6, F7, F8 | Default for unknown/rough/noisy functions. Twice differentiable — realistic assumption for most real-world processes. |
| **RBF** | F5 | Function described as unimodal with a clean peak — consistent with infinite differentiability. Only appropriate when smoothness is well-supported. |

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
| Phase | Appropriate ξ | Reason |
|-------|--------------|--------|
| Weeks 1–2 | 0.05–0.10 | Need to explore; current best is provisional |
| Week 3 | 0.02–0.05 | Converging on best region; reduce jumps |
| Weeks 4+ | 0.005–0.02 | Best neighbourhood confirmed; tight exploitation only |

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

| Fn | All-time best | W4 result | W4 failure mode | W5 action |
|----|--------------|-----------|-----------------|-----------|
| F1 | ≈0 (no signal) | ≈0 | Revisited tested region | Lower-left probe [0.08, 0.48] |
| F2 | **0.648 (W4 NEW)** | 0.648 ↑ | — | Tight continuation near [0.700, 0.961] |
| F3 | −0.018 (W2) | −0.064 | High A=0.983 | Return to [0.439, 0.461, 0.503], ξ=0.02 |
| F4 | −1.177 (W2) | −2.370 | P3=0.499 too high | Lock P3≈0.270, return to W2-like coords |
| F5 | 1374.52 (W3) | 1124.92 | C1/C4 drift with "mean" acq | EI ξ=0.01, return C4→0.872 |
| F6 | −0.384 (W3) | −1.294 | Butter=0.969 catastrophic | "Mean" acq, return to W3 coords |
| F7 | 2.358 (W2) | 0.745 | ξ=0.1 large jump, low lr/reg | Submit W2 best exactly: [0.095, 0.365, ...] |
| F8 | 9.704 (W2) | 8.284 | D3=0.251, D6=0.769 too high | D3<0.07, D6<0.15 hard constraints |

---

## Week 5 → Week 6 Strategic Shift

### What happened in week 6
Week 6 produced two new all-time bests (F2: 0.726, F4: +0.136, F6: −0.296), two confirmed-safe regressions (F3: −0.013, F7: 2.189), and two acquisition-driven failures (F5: 1223, F8: 9.189). F1 continued to return effectively zero.

The dominant pattern in the failures was **UCB over-exploration when hard constraints were not enforced**. F8's acquisition with β=2.5 produced a query at D1=0.471, D4=0.417 — both far outside confirmed safe bounds. F5's D4 reduction from 0.872 to 0.811 violated the D4 > 0.87 hard constraint. Both failures were preventable.

The dominant pattern in the successes was **pure exploitation with GP posterior mean** (F6) and **het-GP tight targeting** (F2). The "mean" acquisition has now produced three consecutive improvements for F6 — the most consistent run of any function. F4's breakthrough to positive Y came from UCB remaining close to the W5 coordinates.

### Key finding: manual constraint enforcement is now mandatory

At this stage of the search (6–7 queries submitted, n = 30–50 including initial data), the GP has enough uncertainty in remote regions to override all prior knowledge if β or ξ is not tightly managed. The solution is to **clip any suggested query outside confirmed bounds before submitting**, regardless of what the acquisition function recommends.

| Acquisition | Appropriate β/ξ | Notes |
|-------------|----------------|-------|
| UCB | β ≤ 1.5 | Reduce from 2.5 — only safe with hard clipping |
| EI | ξ ≤ 0.01 | "Mean" (ξ≈0) is preferred for converged functions |
| Mean | N/A | Default for F6; consider for F3, F5, F7 |

### Direction for week 7

| Fn | All-time best | W6 result | W6 outcome | W7 action |
|----|--------------|-----------|------------|-----------|
| F1 | ≈0 | ≈0 | Cluster exhausted | Abandon cluster — probe [0.08, 0.20] (lower-left) |
| F2 | **0.726 (W6 NEW)** | 0.726 ↑ | New best | [0.699, 0.920] — probe X₂ slightly lower |
| F3 | −0.009 (W5) | −0.013 | Stable near-best | [0.438, 0.462, 0.505] — micro-nudge, ξ=0.005 |
| F4 | **+0.136 (W6 NEW)** | +0.136 ↑ | First positive ever | [0.450, 0.420, 0.365, 0.378] — stay very close, EI ξ=0.01 |
| F5 | 1412.6 (W5) | 1223 | D4 regression | Return to W5 exact: [0.339, 0.838, 0.946, 0.872] |
| F6 | **−0.296 (W6 NEW)** | −0.296 ↑ | New best | [0.410, 0.415, 0.780, 0.785, 0.015] — push Eggs↑, Milk↓ |
| F7 | 2.357 (W2/W5) | 2.189 | D1 test complete | Return to W2/W5 exact: [0.095, 0.365, 0.337, 0.317, 0.362, 0.721] |
| F8 | 9.800 (W5) | 9.189 | Constraint violation | [0.130, 0.235, 0.025, 0.030, 0.985, 0.200, 0.330, 0.720] — enforce all hard limits |

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

| Fn | All-time best | W8 result | W8 outcome | W9 action |
|----|--------------|-----------|------------|-----------|
| F1 | **1.6×10⁻⁷ (W8 NEW)** | 1.6×10⁻⁷ ↑ | Breakthrough | [0.670, 0.695] — toward magnitude centre |
| F2 | 0.726 (W6) | 0.715 | Near-best | [0.699, 0.932] — stay tight |
| F3 | −0.009 (W5) | −0.017 | Drifting | Return to W5 exact: [0.439, 0.461, 0.503] |
| F4 | **+0.367 (W8 NEW)** | +0.367 ↑ | 3rd consecutive | [0.438, 0.431, 0.355, 0.380] — stay tight |
| F5 | **1963.7 (W8 NEW)** | 1963.7 ↑ | Massive jump | [0.350, 0.923, 0.961, 0.880] — push D2 higher |
| F6 | **−0.246 (W8 NEW)** | −0.246 ↑ | Recovered | [0.475, 0.410, 0.740, 0.785, 0.015] — stay tight |
| F7 | **2.377 (W8 NEW)** | 2.377 ↑ | 6-week plateau broken | [0.073, 0.358, 0.341, 0.322, 0.260, 0.727] — test D5 lower |
| F8 | **9.830 (W8 NEW)** | 9.830 ↑ | D3/D4→0 confirmed | [0.080, 0.220, 0.003, 0.015, 0.965, 0.500, 0.326, 0.871] |

---

## Hard Dimension Constraints (Evidence-Based)

These constraints are derived from multiple weeks of observation and should not be violated without strong analytical justification:

| Fn | Constraint | Evidence |
|----|------------|----------|
| F1 | X₁ ∈ [0.65, 0.72]; X₂ ∈ [0.68, 0.73] | W8 at [0.691, 0.707] → 1.6×10⁻⁷; radial decay analysis confirms hotspot centre near [0.65, 0.68] |
| F2 | X₁ ∈ [0.69, 0.71]; X₂ ∈ [0.92, 0.94] | W6/W8 top-2 at X₂=0.932/0.927; W7 regression at X₂=0.921 |
| F3 | A ∈ [0.43, 0.46]; B ∈ [0.46, 0.52]; C ∈ [0.47, 0.52] | W5 best at [0.439, 0.461, 0.503] |
| F4 | All dims ∈ [0.35, 0.45] | Three consecutive bests W6–W8 all within this range |
| F5 | C1 ∈ [0.33, 0.36]; C2 > 0.90; C3 > 0.95; C4 > 0.87 | W8 best at D2=0.915 (+32% vs D2=0.842 in W7) |
| F6 | Flour ∈ [0.40, 0.50]; Eggs ∈ [0.73, 0.77]; Butter ∈ [0.77, 0.83]; Milk < 0.02 | W8 best at [0.472, 0.407, 0.735, 0.782, 0.018] |
| F7 | D1 ∈ [0.07, 0.10]; D2 ∈ [0.355, 0.370]; D5 ∈ [0.26, 0.37]; D6 > 0.72 | W8 best at D1=0.073, D5=0.272 — both lower than W2 |
| F8 | D1 < 0.10; D3 < 0.005; D4 < 0.02; D5 > 0.94 | W8 best at D3=0.004, D4=0.019 — push both toward zero |

---

## Lessons Learned

| Week | Lesson |
|------|--------|
| 1 | **Always integrate all available data before generating suggestions.** The initial .npy data is the most important training signal; ignoring it wastes queries. |
| 1 | **Meta-tracking matters.** Without recording acq/β/ξ/kernel per submission, it is impossible to review why a particular query was made. Now tracked automatically. |
| 1 | **Kernel choice is a one-time structural decision.** RBF was wrong for F1 — changed to Matérn after analysis. Not appropriate to oscillate between kernels. |
| 1 | **Over-exploration in week 1 has downstream cost.** With ~8 remaining queries per function, any query below the initial best is a wasted opportunity. |
| 3–4 | **High ξ (0.05–0.1) is explorative even under EI label.** EI with ξ=0.1 behaves more like UCB than like exploitation in 5–8D with ~30 observations. Reduce ξ aggressively after the best region is confirmed. |
| 4 | **Standardise Y before fitting.** Without standardisation, β and ξ have different real-world meanings for each function. The arcsinh and z-score transforms were the most impactful surrogate improvements made. |
| 4 | **Heteroscedastic GP for explicitly noisy functions.** F2's per-point noise model directly contributed to the new all-time best in W4. Standard GP with constant alpha misread noise as spatial gradient. |
| 4 | **GBM cannot replace GP at current dataset sizes.** GBM achieves LOO R²=0.798 vs GP's 0.969 for F8, and memorises training data (in-sample R²=0.9995). Cross-validation is essential for surrogate model selection — in-sample metrics are meaningless in the BBO context. |
| 4 | **GP+GBM ensemble breaks when GBM overfits.** Residual GP receives near-zero residuals, collapsing its posterior std ≈ 0 → acquisition function explores nothing. Always validate ensemble components individually. |
| 4 | **Deep Ensemble (K=10) underestimates uncertainty.** 95% PI coverage = 0.907 vs target 0.950 for F8 — all K models agree in extrapolation. GP's coverage = 0.977. Use GP for calibrated uncertainty. |
| 6 | **Enforce hard dimension constraints manually.** UCB with β=2.5 will override any implicit boundary knowledge and push queries into known-bad regions. If the acquisition suggests a value outside a confirmed constraint, clip it before submitting. |
| 6 | **Single-dimension perturbation tests have clear ROI.** F7's D1 test (0.095 → 0.013) returned a crisp answer in one query. When the optimum is narrow and deterministic, this protocol is more informative than general EI exploration. |
| 6 | **Cluster exhaustion is a valid stopping criterion.** Six F1 queries in the same region with monotonically decreasing Y confirms the hotspot is elsewhere. Move to a completely new sub-domain rather than continuing to refine within a barren cluster. |
| 6 | **Positive Y for F4 confirms that BBO rediscovers ML tuning intuition.** The "moderate everything" optimum at [0.45, 0.42, 0.36, 0.38] is consistent with regularisation theory — extreme settings on any dimension produce worse models. |
| 6–7 | **Kernel choice is NOT a one-time decision.** F4's Matérn 5/2 achieved LOO R²=0.485 — barely better than a mean predictor. Switching to Matérn 3/2 improved this to 0.961. Kernel selection must be re-evaluated as data accumulates and the landscape is better understood. |
| 6–7 | **Rational Quadratic captures multi-scale structure.** For F7, RQ (0.868) outperformed Matérn 5/2 (0.493–0.722) by modelling both the broad flat landscape and the narrow peak simultaneously. Useful when a function has features at different length-scales. |
| 6–7 | **NGBoost 95% PI coverage is dangerously under-calibrated at small n.** Coverage of 2–25% means the model's uncertainty is meaningless for acquisition. Tree-based probabilistic surrogates require n ≥ 100 to be viable. |
| 6–7 | **When the GP fails, go model-free.** F1's 183-order dynamic range defeats any kernel-based model. Spearman rank correlation on distance (r=−0.696, p=0.002) revealed the hotspot structure that the GP could not detect. |
| 6–7 | **Initial data is a designed experiment — read it first.** The challenge designers placed F1's two highest-magnitude points near [0.65–0.73, 0.68–0.73]. Recognising this as a deliberate bracket of the hotspot should have been the week 1 strategy, not the week 7 one. |
| 8 | **Between-weeks analysis is the highest-ROI activity.** 4 of 6 new bests in W8 were directly enabled by between-weeks engineering (kernel variants, F1 hotspot hunt). Investing time in surrogate diagnostics pays off more than individual query tuning. |
| 8 | **Constraints that were "confirmed" can still be wrong.** F5's D2 was held at [0.83, 0.85] for four weeks based on early data. Releasing it to 0.915 produced a 32% yield jump — the constraint was based on too few observations in the high-D2 region. |
| 8 | **F1 validated: model-free beats model-based when dynamics are extreme.** The radial Spearman analysis predicted the hotspot correctly from 17 points where the GP could not. For extreme-dynamic-range functions, spatial statistics outperform kernel methods. |
