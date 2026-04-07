# Weekly Reflections

One entry per function per week following the challenge format.

---

## Week 1 — 2026-04-04

### Function 1 — 2D Contamination Field

**Submitted:** [0.999247, 0.985647] → **Y = ≈0.000** (1.47×10⁻¹⁸⁵)

**Exploration or exploitation?** Exploration — top-right corner, maximally uncertain region.

**Did it improve on the best?** No. All known outputs remain near zero.

**What did it teach us?** The top-right quadrant (X₁ > 0.7, X₂ > 0.7) contains no signal. The hotspot has not been found. Initial data also all near-zero, confirming the signal is extremely localised.

**Strategy for next week:** Systematic quadrant exploration. Query top-left [0.15, 0.85] — maximally distant from all prior observations. Keep UCB β=2.0 (do not reduce until a non-zero reading is found). Kernel changed from RBF to Matérn 5/2 at end of week 1 — better suited to localised sharp signals.

---

### Function 2 — 2D Noisy Log-Likelihood

**Submitted:** [0.115878, 0.884079] → **Y = 0.0259**

**Exploration or exploitation?** Exploration — low-X₁ region, far from initial best.

**Did it improve on the best?** No. Initial best remains 0.6112 at [0.703, 0.927].

**What did it teach us?** Low-X₁ region is poor. There is a strong dependence on X₁ ≥ 0.7. The top-right quadrant (high X₁, high X₂) is clearly the most promising region.

**Strategy for next week:** Switch to EI ξ=0.01 to exploit near [0.703, 0.927]. Submit near [0.75, 0.93] — slight X₁ increase to probe whether peak extends further right. Reduce β if staying with UCB.

---

### Function 3 — 3D Drug Compounds

**Submitted:** [0.778655, 0.249003, 0.41851] → **Y = -0.0417**

**Exploration or exploitation?** Exploration — moved away from best, tested high-A low-B region.

**Did it improve on the best?** No. Initial best remains -0.0348 at [0.490, 0.610, 0.340].

**What did it teach us?** Low Compound B (0.249) is worse than the initial best's B≈0.61. This validates that Compound B in the 0.55–0.65 range is important. High Compound A alone is not sufficient.

**Strategy for next week:** EI ξ=0.01. Submit near [0.48, 0.65, 0.32] — close to initial best, nudge B higher, reduce C slightly. Do not abandon the initial best neighbourhood.

---

### Function 4 — 4D Warehouse Hyperparameters

**Submitted:** [0.438110, 0.032583, 0.981555, 0.372065] → **Y = -21.254**

**Exploration or exploitation?** Exploration — extreme values in P2 and P3.

**Did it improve on the best?** No. Initial best remains -4.0255.

**What did it teach us?** Extreme input values (P2≈0, P3≈1) are clearly suboptimal. The best region is mid-range across all dimensions. Low P4 appears beneficial (initial best has P4≈0.25; week 1 used 0.37 and scored worse).

**Strategy for next week:** Switch to EI ξ=0.05. Target [0.55, 0.38, 0.50, 0.18] — exploiting initial best neighbourhood with a push toward lower P4.

---

### Function 5 — 4D Chemical Yield

**Submitted:** [0.816816, 0.085090, 0.386806, 0.716711] → **Y = 50.44**

**Exploration or exploitation?** Exploration — high C1, low C2, mid C3/C4.

**Did it improve on the best?** No. Initial best remains 1088.86 — week 1 result is 21× lower.

**What did it teach us?** The initial best at [0.224, 0.846, 0.879, 0.879] clearly dominates. High C1 / low C2 is strongly suboptimal. The unimodal landscape means the peak is well-defined and near the initial best.

**Strategy for next week:** EI ξ=0.01. Target [0.18, 0.87, 0.90, 0.90] — push C1 lower, push C2/C3/C4 higher. Every remaining query should be within ±0.05 of [0.22, 0.85, 0.88, 0.88].

---

### Function 6 — 5D Cake Recipe

**Submitted:** [0.834917, 0.531385, 0.123811, 0.512032, 0.474771] → **Y = -1.8257**

**Exploration or exploitation?** Exploration — high Sugar (dim 2) and high Milk (dim 5) vs initial best.

**Did it improve on the best?** No. Initial best remains -0.7143 at [0.728, 0.155, 0.733, 0.694, 0.056].

**What did it teach us?** Elevated Sugar (0.531 vs 0.155) and Milk (0.475 vs 0.056) are strongly penalising. The best recipe has low Sugar and very low Milk. High Flour, Eggs and Butter appear beneficial.

**Strategy for next week:** EI ξ=0.01. Target [0.72, 0.10, 0.75, 0.72, 0.04] — anchor near initial best with further reductions in Sugar and Milk. Hard constraint: Sugar < 0.20, Milk < 0.10 in all future queries.

---

### Function 7 — 6D GBM Hyperparameters

**Submitted:** [0.333, 0.310, 0.250, 0.800, 0.800, 0.050] → **Y = 0.1207**

**Exploration or exploitation?** Informed start heuristic — domain-guided initial query.

**Did it improve on the best?** No. Initial best remains 1.365 at [0.058, 0.49, 0.25, 0.22, 0.42, 0.73].

**What did it teach us?** The informed start moved away from the initial best in 4 of 6 dimensions (high subsample/max_features, low regularisation vs initial best which is low subsample/mid max_features/high regularisation). The GBM landscape favours low n_estimators, moderate learning_rate, low subsample and high regularisation.

**Strategy for next week:** EI ξ=0.01. Perturb around initial best [0.058, 0.49, 0.25, 0.22, 0.42, 0.73]. Try pushing regularisation higher (dim6 0.73 → 0.80) and exploring the low n_estimators / moderate learning_rate trade-off.

---

### Function 8 — 8D ML Hyperparameters

**Submitted:** [0.241537, 0.754347, 0.171428, 0.086557, 0.351128, 0.974082, 0.194979, 0.655785] → **Y = 9.2597**

**Exploration or exploitation?** Mixed — GP-guided query in 8D space.

**Did it improve on the best?** No. Initial best remains 9.5985. Week 1 is close but did not beat it.

**What did it teach us?** Very low P1–P4 with high P6 and high P8 is the promising pattern. Week 1 used higher P1-P4 values and lower P8 than the initial best — both appear detrimental.

**Strategy for next week:** UCB β=2.5. Perturb initial best toward even lower P1-P4. Target [0.03, 0.04, 0.05, 0.02, 0.38, 0.85, 0.50, 0.92].

---

---

## Week 2 — 2026-04-07

### Function 1 — 2D Contamination Field

**Submitted:** [0.855900, 0.278400] → **Y ≈ 0.000** (effectively zero)

**Acquisition:** UCB β=2.0, Matérn 5/2

**Exploration or exploitation?** Exploration — deliberately moved to the lower-right quadrant to test an unsampled region.

**Did it improve on the best?** No. All observations remain effectively zero. No signal has been detected in any of the four quadrants tested so far.

**What did it teach us?** The lower-right quadrant (X₁ > 0.5, X₂ < 0.5) also contains no signal. Three of four quadrants have now been tested — top-right (week 1), lower-right (week 2), and indirectly through initial data. The signal is either extremely localised, concentrated in the top-left or lower-left quadrant, or requires a very specific combination of X₁ and X₂ values we have not yet hit.

**Strategy for next week:** Target the top-left quadrant — [0.15, 0.85]. This is the only major region not yet tested. If that also returns near-zero, the function may have a single isolated hotspot requiring very dense grid-style search rather than GP guidance. Maintain UCB β=2.0 to keep exploration pressure high.

---

### Function 2 — 2D Noisy Log-Likelihood

**Submitted:** [0.815442, 0.962028] → **Y = 0.0528**

**Acquisition:** UCB β=2.5, Matérn 5/2

**Exploration or exploitation?** Exploitation attempt — targeted the high-X₁ / high-X₂ region near the initial best.

**Did it improve on the best?** Improved on week 1 (0.026 → 0.053) but still far below the initial best of 0.6112. The initial best at [0.703, 0.927] remains the all-time best.

**What did it teach us?** Moving X₁ higher (0.703 → 0.815) made things worse. The initial best appears to sit in a narrow band around X₁ ≈ 0.70 rather than extending toward X₁ ≈ 1.0. The SVR analysis in `analysis/02_function2_svr_exploration.ipynb` confirms both GP and SVR-RBF agree the next query should be near [0.70, 0.93] — directly at the initial best neighbourhood rather than to the right of it. The function is noisy with local optima; week 2 likely caught a noise trough.

**Strategy for next week:** Submit directly at [0.697, 0.933] — the GP UCB suggestion that converges on the initial best neighbourhood. Stop moving X₁ rightward. Switch to EI ξ=0.01 to focus exploitation on the known high-value region.

---

### Function 3 — 3D Drug Compounds

**Submitted:** [0.445600, 0.338900, 0.486400] → **Y = -0.0182**

**Acquisition:** EI β=1.96, ξ=0.02, Matérn 5/2

**Exploration or exploitation?** Mixed — EI guided a query that moved toward lower Compound B than the initial best.

**Did it improve on the best?** **Yes — new all-time best.** Improved from -0.0348 (initial) to -0.0182. An improvement of 0.0166, reducing adverse reactions by approximately 48%.

**What did it teach us?** This result challenges the week 1 hypothesis that Compound B should be high (0.61). Week 2 achieved a better outcome at B=0.339 — roughly half the week 1 assumption. This suggests either: (a) lower B is genuinely better, or (b) the interaction between A, B, and C at these levels is more important than any single dimension. Compound A at 0.446 and C at 0.486 both differ substantially from the initial best too — the GP found a different region that outperformed the initial data.

**Strategy for next week:** Exploit the new best neighbourhood. EI ξ=0.01 to tighten around [0.45, 0.34, 0.49]. Also probe whether B can be reduced further — try [0.43, 0.28, 0.50] as a secondary direction.

---

### Function 4 — 4D Warehouse Hyperparameters

**Submitted:** [0.459600, 0.413400, 0.311100, 0.404700] → **Y = -1.1765**

**Acquisition:** UCB β=2.0, ξ=0.05, Matérn 5/2

**Exploration or exploitation?** Exploitation — mid-range values across all four dimensions, anchored near the initial best neighbourhood.

**Did it improve on the best?** **Yes — new all-time best by a large margin.** Improved from -4.0255 (initial) to -1.1765. A 70.8% reduction in the negative gap. This is the single largest improvement across all functions this week.

**What did it teach us?** Mid-range values (0.31–0.46 across all four parameters) dramatically outperform the extreme values tested in week 1. The warehouse model performs best with balanced rather than extreme parameter settings. The GP had a good model of the landscape — the UCB suggestion correctly identified this mid-range cluster as the high-value zone. The landscape appears to have a broad, relatively smooth peak in the central region of the 4D space.

**Strategy for next week:** Switch to EI ξ=0.01 to tighten exploitation around [0.46, 0.41, 0.31, 0.40]. Explore whether any single dimension can be pushed to improve further. Try a slight push in all four dimensions toward the centre: [0.50, 0.42, 0.33, 0.42].

---

### Function 5 — 4D Chemical Yield

**Submitted:** [0.284200, 0.834600, 0.909800, 0.865500] → **Y = 1138.865**

**Acquisition:** EI β=1.5, ξ=0.01, RBF kernel

**Exploration or exploitation?** Exploitation — closely followed the initial best neighbourhood, shifting C1 slightly higher and C2/C3/C4 higher.

**Did it improve on the best?** **Yes — new all-time best.** Improved from 1088.86 (initial) to 1138.87 (+4.6%). The function is unimodal and the peak is clearly in this region.

**What did it teach us?** The RBF kernel and low-ξ EI correctly concentrated the search near the initial best. The slight upward adjustment in C1 (0.224 → 0.284) combined with higher C3/C4 (0.879 → 0.910/0.866) produced the improvement. The unimodal structure means continued close exploitation is the correct strategy — there are no competing peaks to be found elsewhere.

**Strategy for next week:** Continue tight exploitation via EI ξ=0.01. Submit near [0.265, 0.845, 0.920, 0.875] — a small step from the week 2 best, probing whether the peak extends slightly toward higher C3 and higher C4. Do not increase C1 further; the correlation evidence suggests C1 should remain low.

---

### Function 6 — 5D Cake Recipe

**Submitted:** [0.446300, 0.261000, 0.435100, 0.718500, 0.162300] → **Y = -0.5178**

**Acquisition:** EI β=1.96, ξ=0.02, Matérn 5/2

**Exploration or exploitation?** Exploitation — EI targeted a region somewhat distant from the initial best but in a direction the GP assessed as promising.

**Did it improve on the best?** **Yes — new all-time best.** Improved from -0.7143 (initial) to -0.5178 (+27.5% reduction in penalty score).

**What did it teach us?** The winning recipe differs noticeably from the initial best: Flour lower (0.446 vs 0.728), Sugar higher (0.261 vs 0.155), Eggs lower (0.435 vs 0.733), Butter similar (0.719 vs 0.694), Milk higher (0.162 vs 0.056). This challenges the week 1 hypothesis that Milk and Sugar must be minimised. A balanced ingredient profile appears better than the initially assumed extreme-low-sugar, extreme-low-milk configuration. The GP found a genuinely better region by exploring more broadly.

**Strategy for next week:** Exploit the new best neighbourhood. EI ξ=0.01. Target [0.44, 0.26, 0.44, 0.72, 0.16] — very close to week 2 best. Also test whether reducing Eggs slightly further and increasing Flour modestly helps.

---

### Function 7 — 6D GBM Hyperparameters

**Submitted:** [0.094700, 0.365000, 0.337000, 0.317000, 0.361500, 0.720800] → **Y = 2.3576**

**Acquisition:** EI β=1.96, ξ=0.05, Matérn 5/2

**Exploration or exploitation?** Exploitation — EI targeted the neighbourhood of the initial best with adjusted parameters.

**Did it improve on the best?** **Yes — new all-time best by a large margin.** Improved from 1.3650 (initial) to 2.3576 (+72.6%). The largest relative gain across all functions this week.

**What did it teach us?** Very low n_estimators (dim1=0.095) combined with moderate learning_rate (0.365) and high regularisation (dim6=0.721) is the winning combination. The GBM model benefits from being kept small and heavily regularised — consistent with avoiding overfitting in the underlying model being tuned. The subsample (0.317) and max_features (0.362) both being below 0.4 suggests that stochastic subsampling with limited features is beneficial. The EI acquisition function correctly identified this as an improvement over the initial best.

**Strategy for next week:** Tighten exploitation with EI ξ=0.01. Target [0.08, 0.37, 0.33, 0.32, 0.36, 0.75] — push n_estimators even lower and regularisation slightly higher. The Bayesian evidence strongly points to this low n_estimators / high regularisation combination being the peak region.

---

### Function 8 — 8D ML Hyperparameters

**Submitted:** [0.211707, 0.204113, 0.040404, 0.040332, 0.972678, 0.066676, 0.219298, 0.061115] → **Y = 9.7035**

**Acquisition:** UCB β=2.5, ξ=0.1, Matérn 5/2

**Exploration or exploitation?** Exploitation — UCB β=2.5 directed a query with very low D1–D4, very high D5, and low D6–D8.

**Did it improve on the best?** **Yes — new all-time best.** Improved from 9.5985 (initial) to 9.7035 (+1.1%).

**What did it teach us?** The correlation analysis from `analysis/03_function8_rf_surrogate.ipynb` is confirmed: Dims 1, 3, and 7 must be kept LOW (r ≈ −0.65, −0.66, −0.37). The winning input has D1=0.21, D2=0.20, D3=0.04, D4=0.04, D5=0.97, D6=0.07, D7=0.22, D8=0.06. Notable pattern: D3 and D4 near zero, D5 near maximum (0.97) is a distinctive signature. The RF surrogate's feature importance independently confirms D3 and D1 as the dominant dimensions.

**Strategy for next week:** Week 3 result pending. Once received, rerun `analysis/03_function8_rf_surrogate.ipynb` with `WEEK3_Y` set. Current GP UCB suggestion is [0.044, 0.014, 0.325, 0.074, 0.804, 0.997, 0.061, 0.433] — predicted Y=9.84. D6 pushed high (0.997) is atypical; validate against RF cross-check before submitting.

---

---

## Week 3 — 2026-04-14

### Function 1 — 2D Contamination Field

**Submitted:** [0.150000, 0.500000] → **Y ≈ 0.000**

**Acquisition:** UCB β=2.0, Matérn 5/2, arcsinh Y-transform

**Exploration or exploitation?** Exploration — targeted the left half of the space (X₁=0.15), mid-height.

**Did it improve on the best?** No. Four distinct regions have now returned near-zero: top-right (W1), lower-right (W2), left-mid (W3), and the initial data cluster. No signal has been detected anywhere.

**What did it teach us?** The contamination hotspot has not been found in three weeks. The arcsinh Y-transform was enabled this week, which spreads the near-zero values to give the GP more structure to learn from. However without any non-zero reading the transform cannot help the acquisition function converge. The function may have a single extremely narrow peak that requires either very precise targeting or substantially more systematic search. The initial data's "best" Y = 7.71×10⁻¹⁶ is effectively numerical noise rather than a real signal.

**Strategy for next week:** Abandon GP guidance temporarily — it has nothing useful to learn from. Switch to systematic grid-based exploration of the lower-left quadrant [X₁ < 0.5, X₂ < 0.5], which is the only major region not yet tested. Submit [0.25, 0.25] as a representative point. If still zero, consider [0.50, 0.15] (bottom-centre) in week 5.

---

### Function 2 — 2D Noisy Log-Likelihood

**Submitted:** [0.693500, 0.905800] → **Y = 0.4929**

**Acquisition:** UCB β=2.5, Matérn 5/2

**Exploration or exploitation?** Exploitation — closely targeted the initial best region [0.703, 0.927].

**Did it improve on the best?** Significant improvement on portal submissions (0.053 → 0.493 — a 9× jump). However, still below the initial best of 0.6112. The all-time best remains in the initial data.

**What did it teach us?** Targeting the initial best neighbourhood directly worked — 0.493 is close to the initial best of 0.611 and shows the GP surrogate has now learned the right direction. Week 1 and 2's lower scores were caused by querying away from this region. The SVR analysis confirmed this region as the consensus peak. The remaining gap (0.493 vs 0.611) may be noise in the function or may indicate the true peak is slightly displaced from the initial best point.

**Strategy for next week:** Tighten exploitation further. Submit at the GP posterior mean maximiser — use "mean" acquisition or EI ξ=0.001. Target very close to [0.703, 0.927] — within ±0.02 of the initial best coordinates. The function is noisy so multiple nearby queries will eventually bracket the true peak.

---

### Function 3 — 3D Drug Compounds

**Submitted:** [0.558800, 0.030300, 0.554200] → **Y = -0.1227**

**Acquisition:** EI β=1.96, ξ=0.02, Matérn 5/2

**Exploration or exploitation?** Exploration — EI moved the query to a very different region, particularly Compound B ≈ 0.03 (extremely low).

**Did it improve on the best?** No — severe regression. Dropped from -0.0182 (W2 best) to -0.1227. Worst result across all three weeks, substantially worse than even the initial best (-0.0348).

**What did it teach us?** Compound B near zero (0.030) is clearly catastrophic. This strongly re-establishes the importance of Compound B being in a reasonable range. Week 2 had B=0.339 and produced the best result; week 3's B=0.030 produced the worst. The EI acquisition over-explored here — moving 5× away from the current best in B-space. The lesson is that with only 18 observations in 3D, EI can generate large exploratory jumps that are counterproductive. We should tighten the acquisition to stay near the current best.

**Strategy for next week:** Return to tight exploitation around the W2 best [0.446, 0.339, 0.486]. Switch to EI ξ=0.005 or "mean" acquisition. Enforce a soft constraint: do not submit with Compound B < 0.20. The GP needs to see more observations around the known good region before it can reliably guide exploration.

---

### Function 4 — 4D Warehouse Hyperparameters

**Submitted:** [0.455700, 0.406300, 0.384700, 0.304300] → **Y = -1.5683**

**Acquisition:** UCB β=2.0, ξ=0.05, Matérn 5/2, ARD enabled

**Exploration or exploitation?** Exploitation — close to W2 best with slight parameter adjustments.

**Did it improve on the best?** No. Slight regression from -1.1765 (W2) to -1.5683 (-33%). The W2 best at [0.460, 0.413, 0.311, 0.405] remains the all-time best.

**What did it teach us?** The peak is narrow in this 4D space — small perturbations from the W2 best coordinates produced a worse result. The shift in P3 (0.311 → 0.385) and P4 (0.405 → 0.304) relative to W2 appear to have caused the regression. ARD was enabled this week; with more data it should help identify which dimensions are critical. P3 has the weakest correlation with Y (r=−0.16) but moving it from 0.311 to 0.385 still hurt — suggesting there is a local interaction effect around the current best region.

**Strategy for next week:** Return to near the W2 best: [0.46, 0.41, 0.31, 0.41]. Reduce ξ to 0.01 — exploit more tightly. The landscape around the current best is sharp; do not deviate more than ±0.05 per dimension from W2 coordinates.

---

### Function 5 — 4D Chemical Yield

**Submitted:** [0.361600, 0.837400, 0.938600, 0.872400] → **Y = 1374.524**

**Acquisition:** EI β=1.5, ξ=0.01, RBF kernel

**Exploration or exploitation?** Exploitation — tight follow-on from W2 best.

**Did it improve on the best?** **Yes — new all-time best.** Three consecutive weekly improvements: 50.44 → 1138.87 → 1374.52. Total gain from initial best: +26.2%.

**What did it teach us?** The unimodal structure continues to reward tight exploitation. C1 has drifted slightly higher (0.224 → 0.284 → 0.362) across the three weeks while C3/C4 remain high (0.91/0.87). The peak may lie at slightly higher C1 than the initial best suggested, or the EI is finding a slightly different cross-section of a broad peak. Either way, the strategy is working — each week's result is better than the last.

**Strategy for next week:** Switch to "mean" acquisition (pure exploitation, no exploration bonus). The unimodal structure is confirmed with high confidence. Submit near [0.38, 0.84, 0.94, 0.88] — continuing the gentle C1 upward drift while keeping C3/C4 high. Consider whether C2 can be nudged (currently stable at 0.837).

---

### Function 6 — 5D Cake Recipe

**Submitted:** [0.380200, 0.480600, 0.559600, 0.724800, 0.169500] → **Y = -0.3837**

**Acquisition:** EI β=1.96, ξ=0.02, Matérn 5/2

**Exploration or exploitation?** Mixed — EI moved substantially in Flour (0.446 → 0.380) and Eggs (0.435 → 0.560) relative to W2.

**Did it improve on the best?** **Yes — new all-time best.** Two consecutive portal improvements: -0.518 → -0.384 (+26%). Now 46% better than the initial best (-0.714).

**What did it teach us?** The recipe landscape continues to reward a specific combination: moderate Flour (~0.38–0.45), moderate Sugar (~0.26–0.48), higher Eggs (~0.44–0.56), high Butter (~0.72), and low-moderate Milk (~0.16–0.17). The initial hypothesis that Sugar and Milk must be minimised has been contradicted — moderate Sugar (0.48 this week) still produced the best result. The GP is converging on a genuine peak region that is distinct from the initial best.

**Strategy for next week:** Exploit tightly around W3 best. EI ξ=0.01. Target [0.37, 0.47, 0.57, 0.73, 0.17] — marginal increments in the improving directions. Butter and Milk appear stable; focus on whether Flour can be reduced further and Eggs raised slightly.

---

### Function 7 — 6D GBM Hyperparameters

**Submitted:** [0.061100, 0.427900, 0.284300, 0.256300, 0.374400, 0.724200] → **Y = 1.9315**

**Acquisition:** EI β=1.96, ξ=0.05, Matérn 5/2, ARD enabled

**Exploration or exploitation?** Exploitation — similar to W2 best with modest adjustments.

**Did it improve on the best?** No. Regression from 2.358 (W2) to 1.931 (−18%). The W2 submission [0.095, 0.365, 0.337, 0.317, 0.362, 0.721] remains the all-time best.

**What did it teach us?** The primary difference between W2 and W3 is dim1 (n_estimators): 0.095 → 0.061 (lower in W3) and dim2 (learning_rate): 0.365 → 0.428 (higher in W3). The regression suggests the W2 combination sits in a narrow sweet spot — moving to higher learning rate and lower n_estimators simultaneously pushed outside the optimal region. Dim6 (regularisation) was nearly identical (0.721 vs 0.724), confirming that dimension is well-calibrated. ARD should be learning that dim1 and dim2 have short length-scales (sensitive) while dim5 is less critical.

**Strategy for next week:** Return to near W2 best: [0.095, 0.365, 0.337, 0.317, 0.362, 0.721]. Reduce ξ to 0.01 to tighten exploitation. Try a tiny perturbation — only one dimension at a time from the known best: push dim1 slightly (0.095 → 0.08) to test whether even lower n_estimators helps.

---

### Function 8 — 8D ML Hyperparameters

**Submitted:** [0.062400, 0.956200, 0.000800, 0.908300, 0.526900, 0.001700, 0.943300, 0.227600] → **Y = 7.3179**

**Acquisition:** UCB β=3.5, ξ=0.1, Matérn 5/2, ARD enabled

**Exploration or exploitation?** Aggressive exploration — β=3.5 pushed the GP UCB suggestion into a very different region. D2=0.956, D4=0.908, D6=0.002, D7=0.943 are all extreme values far from the W2 best.

**Did it improve on the best?** No — significant regression. Dropped from 9.704 (W2) to 7.318 (−24.6%). Worst portal result for this function.

**What did it teach us?** The aggressive exploration at β=3.5 was a mistake at this stage. The correlation analysis is unambiguous: D1 and D3 must be LOW, and W3's D1=0.062 and D3=0.001 respected this — but D2=0.956 and D4=0.908 being simultaneously very high was clearly wrong. The W2 best had D2=0.204 and D4=0.040, confirming that low D2 and D4 are also important. The RF surrogate notebook (`analysis/03_function8_rf_surrogate.ipynb`) should now be re-run with W3 included to reassess the updated correlation and feature importance rankings.

**Strategy for next week:** Reduce β to 2.0 and switch to EI ξ=0.05. Exploit near the W2 best: [0.21, 0.20, 0.04, 0.04, 0.97, 0.07, 0.22, 0.06]. The updated RF notebook will provide the week 4 suggestion. Key constraints: D1 < 0.25, D2 < 0.30, D3 < 0.10, D4 < 0.10. Do not repeat the high-D2/D4 exploration.

---

*Reflections for subsequent weeks will be appended below.*
