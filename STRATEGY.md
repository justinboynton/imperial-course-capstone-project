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

## Hard Dimension Constraints (Evidence-Based)

These constraints are derived from multiple weeks of observation and should not be violated without strong analytical justification:

| Fn | Constraint | Evidence |
|----|------------|----------|
| F2 | X₁ ∈ [0.68, 0.72] | All high-Y observations within this band |
| F3 | Compound A < 0.70; Compound B ∈ [0.25, 0.55] | A=0.983 (W4) → −0.064; B=0.030 (W3) → −0.123 |
| F4 | P3 ∈ [0.28, 0.35] | P3=0.499 → −2.37; P3=0.311 → −1.177 (best) |
| F5 | C1 ∈ [0.28, 0.42]; C4 > 0.85 | C4=0.797 (W4) → 1124 vs C4=0.872 (W3) → 1374 |
| F6 | Butter ∈ [0.70, 0.78]; Eggs > 0.45; Milk < 0.25 | Butter=0.969 (W4) → −1.294 |
| F7 | n_est (D1) < 0.12; lr (D2) ∈ [0.33, 0.42]; reg (D6) > 0.68 | W2 best anchors all three |
| F8 | D1 < 0.25; D3 < 0.07; D4 < 0.08; D5 > 0.90 | GP ARD + GBM permutation importance agree |

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
