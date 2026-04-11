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

**WeekStrategy for next week:** Continue tight exploitation via EI ξ=0.01. Submit near [0.265, 0.845, 0.920, 0.875] — a small step from the week 2 best, probing whether the peak extends slightly toward higher C3 and higher C4. Do not increase C1 further; the correlation evidence suggests C1 should remain low.

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

**What did it teach us?** The recipe landscape continues to reward a specific combination: moderate Flour (~~0.38–0.45), moderate Sugar (~~0.26–0.48), higher Eggs (~~0.44–0.56), high Butter (~~0.72), and low-moderate Milk (~0.16–0.17). The initial hypothesis that Sugar and Milk must be minimised has been contradicted — moderate Sugar (0.48 this week) still produced the best result. The GP is converging on a genuine peak region that is distinct from the initial best.

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

---

## Week 4 — 2026-04-09

### Function 1 — 2D Contamination Field

**Submitted:** [0.774940, 0.763411] → **Y ≈ 0.000**

**Acquisition:** UCB β=1.0, Matérn 5/2, arcsinh Y-transform

**Exploration or exploitation?** Exploration — upper-right quadrant again, despite week 3 strategy recommending the lower-left.

**Did it improve on the best?** No. Five weeks, five queries, zero signal detected in any of them.

**What did it teach us?** W4 accidentally revisited the upper-right region tested in week 1 — the β=1.0 GP suggestion was pulled toward the highest-uncertainty edge of a region already confirmed empty. With all four quadrants now sampled at least once (upper-right W1, lower-right W2, left-mid W3, upper-right again W4), the hotspot either sits in a very narrow band in the lower-left or requires precise coordinates rather than quadrant-level targeting. The GP has nothing useful to learn from five near-identical near-zero outputs; it cannot help locate a signal it has never seen.

**Strategy for next week (W5):** Manual lower-left probe at [0.08, 0.48] — far from all prior queries. If this also returns near-zero, the remaining weeks should focus on a systematic diagonal sweep: [0.15, 0.15], [0.30, 0.70], [0.70, 0.30] to try to bracket the hotspot geometrically.

---

### Function 2 — 2D Noisy Log-Likelihood

**Submitted:** [0.699929, 0.961372] → **Y = 0.6485**

**Acquisition:** UCB β=1.5, Matérn 5/2, heteroscedastic GP, standardised Y

**Exploration or exploitation?** Tight exploitation — almost identical X₁ to the initial best (0.700 vs 0.703), X₂ pushed slightly higher (0.961 vs 0.927).

**Did it improve on the best?** **Yes — new all-time best.** 0.6485 beats the initial data best of 0.6112 (+6.1%). First time any portal submission has beaten the initial data for this function.

**What did it teach us?** The heteroscedastic GP introduced between W3 and W4 played a direct role here. By assigning higher noise to the two conflicting peak-region observations (0.611 and 0.493), the GP stopped treating their gap as a definitive gradient and instead kept its uncertainty wide around [0.70, 0.93]. The UCB suggestion consequently stayed in the right neighbourhood rather than drifting away from the apparent "downhill" direction. X₂=0.961 (slightly higher than the initial best's 0.927) and X₁≈0.700 (unchanged) produced the gain — consistent with the true peak being slightly higher in X₂ than the initial observation suggested.

**Strategy for next week (W5):** Continue tight exploitation. Submit at [0.698, 0.942] — the W5 GP suggestion. Keep UCB β=2.5 (wider uncertainty = better UCB suggestions for a noisy function). The peak is narrow; stay within ±0.02 of [0.700, 0.955].

---

### Function 3 — 3D Drug Compounds

**Submitted:** [0.983498, 0.399509, 0.608719] → **Y = −0.0640**

**Acquisition:** EI β=1.96, ξ=0.05, Matérn 5/2, standardised Y

**Exploration or exploitation?** Unexpected exploration — EI with ξ=0.05 pushed Compound A to 0.983, far from the W2 best neighbourhood.

**Did it improve on the best?** No. Partial recovery from the catastrophic W3 result (−0.1227 → −0.0640), but still well below the W2 all-time best of −0.0182. The W2 coordinates [0.446, 0.339, 0.486] remain the best.

**What did it teach us?** Compound A at 0.983 (extremely high) did not produce a good result — reinforcing that the high-A region is not optimal. The EI with ξ=0.05 is still generating large jumps rather than tight exploitation; the GP landscape is uncertain enough in 3D at n=19 that EI frequently identifies high-uncertainty remote regions as worth exploring. The recovery from W3 confirms Compound B at 0.400 is reasonable (much better than W3's B=0.030), but the W2 level of B≈0.34 remains the best known.

**Strategy for next week (W5):** Return to tight exploitation around the W2 best. Lower ξ to 0.02 to prevent further long-range jumps. The W5 suggestion is [0.439, 0.461, 0.503] — close to the W2 best, reasonable in all three dimensions. Hard constraint: Compound A < 0.60, Compound B in [0.25, 0.55].

---

### Function 4 — 4D Warehouse Hyperparameters

**Submitted:** [0.433271, 0.396639, 0.499133, 0.420139] → **Y = −2.3702**

**Acquisition:** UCB β=1.2, ξ=0.05, Matérn 5/2, standardised Y

**Exploration or exploitation?** Exploitation attempt — mid-range values, but P3 at 0.499 is substantially higher than the W2 best's P3=0.311.

**Did it improve on the best?** No — continued regression for the second consecutive week. W2: −1.177 → W3: −1.568 → W4: −2.370. The W2 best at [0.460, 0.413, 0.311, 0.405] is now three weeks old and still unchallenged.

**What did it teach us?** P3 is the culprit. Moving P3 from 0.311 (W2 best) to 0.385 (W3) produced a −33% regression; moving it to 0.499 (W4) produced a further −51% regression. The function has a sharp, narrow optimum in the P3 dimension — P3 must stay near 0.31. The other three parameters were reasonable in W4 (P1=0.433, P2=0.397, P4=0.420 are all close to the W2 best), confirming P3 is the dominant sensitivity. ARD should be learning this.

**Strategy for next week (W5):** Lock P3 at 0.270 (slightly lower than W2's 0.311, as the trend suggests the true optimum may be slightly below 0.31). Submit at [0.464, 0.420, 0.270, 0.389] — the W5 GP suggestion. Do not deviate P3 above 0.35 in any future submission.

---

### Function 5 — 4D Chemical Yield

**Submitted:** [0.414879, 0.859494, 0.918916, 0.797250] → **Y = 1124.92**

**Acquisition:** Mean (pure exploitation), RBF kernel, standardised Y

**Exploration or exploitation?** Pure exploitation — no exploration bonus, GP posterior mean maximiser.

**Did it improve on the best?** No — regression from the W3 best of 1374.52 to 1124.92 (−18%). First time this function has regressed since week 1.

**What did it teach us?** Switching to "mean" acquisition did not cause the regression — the coordinates drifted. W4 had C1=0.415 (up from W3's 0.362) and C4=0.797 (down from W3's 0.872). The W3 best had C4=0.872; dropping to C4=0.797 is a −8.5% reduction in a dimension that appears sensitive. The GP posterior mean maximiser faithfully followed the GP's estimated peak, but the GP's confidence in the C1/C4 values may have been misleading. The unimodal structure is confirmed but the peak coordinates are narrower than they appeared.

**Strategy for next week (W5):** Switch back to EI ξ=0.01 to let the GP self-correct. The W5 suggestion is [0.339, 0.838, 0.946, 0.872] — pulling C1 back down and restoring C4 to 0.872. This should recover toward the W3 best.

---

### Function 6 — 5D Cake Recipe

**Submitted:** [0.348495, 0.500810, 0.287744, 0.969433, 0.102729] → **Y = −1.2939**

**Acquisition:** EI β=1.5, ξ=0.05, Matérn 5/2, standardised Y

**Exploration or exploitation?** Over-exploration — Butter (dim4) jumped to 0.969, far above the W3 best value of 0.725. Eggs (dim3) dropped sharply to 0.288 from W3's 0.560.

**Did it improve on the best?** No — severe regression. W3 best was −0.384 (new all-time best at the time); W4 returned −1.294. The two consecutive improving weeks (W2: −0.518, W3: −0.384) have been undone.

**What did it teach us?** Butter at 0.969 is catastrophically high — nearly twice the W3 best value of 0.725. Similarly, Eggs at 0.288 vs W3's 0.560 represents a −49% deviation in a dimension that had been steadily improving. The EI with ξ=0.05 on a 5D surface with only 25 observations generated an extremely off-target suggestion. The lesson: for F6, Butter must remain in [0.70, 0.76] and Eggs must stay above 0.45 — these are now confirmed as tight constraints based on the regression evidence.

**Strategy for next week (W5):** Return directly to the W3 best neighbourhood. The W5 GP suggestion is [0.343, 0.523, 0.603, 0.751, 0.141] — very close to W3 best [0.380, 0.481, 0.560, 0.725, 0.170]. Switch to "mean" acquisition to prevent further large jumps. Hard constraints: Butter ∈ [0.70, 0.78], Eggs ∈ [0.45, 0.65], Milk < 0.25.

---

### Function 7 — 6D GBM Hyperparameters

**Submitted:** [0.024247, 0.109602, 0.273023, 0.338380, 0.357029, 0.537200] → **Y = 0.7445**

**Acquisition:** EI β=1.96, ξ=0.1, Matérn 5/2, ARD, standardised Y

**Exploration or exploitation?** Aggressive exploration — very large deviation from W2 best in two critical dimensions: learning_rate dropped to 0.110 (from 0.365), regularisation (dim6) dropped to 0.537 (from 0.721).

**Did it improve on the best?** No — catastrophic regression. W2 best was 2.358; W4 returned 0.745, the second worst result ever on this function. Two weeks of regression (W3: 1.931, W4: 0.745) since the W2 peak.

**What did it teach us?** EI with ξ=0.1 is far too explorative for a 6D function at this stage. The acquisition jumped to a region with learning_rate=0.110 — less than a third of the W2 optimal value of 0.365. With only 33 observations across 6 dimensions, the GP has high uncertainty in unexplored regions, and a high-ξ EI will always find those regions attractive regardless of whether they are genuinely promising. The W2 result has now been confirmed as a genuine peak rather than noise — no subsequent query has come close to matching it.

**Strategy for next week (W5):** Submit directly at the W2 best: [0.095, 0.365, 0.337, 0.317, 0.362, 0.721]. The W5 suggestion [0.095, 0.365, 0.337, 0.317, 0.362, 0.721] is essentially the W2 best reproduced — a sensible choice. Reduce ξ to 0.05 and switch to EI or "mean". The W2 best must be matched before any further deviations are tested.

---

### Function 8 — 8D ML Hyperparameters

**Submitted:** [0.091322, 0.141133, 0.251186, 0.059089, 0.816917, 0.769123, 0.129731, 0.088824] → **Y = 8.2840**

**Acquisition:** UCB β=2.5, ξ=0.1, Matérn 5/2, ARD, standardised Y

**Exploration or exploitation?** Mixed — D3=0.251 (higher than W2's 0.040), D6=0.769 (far higher than W2's 0.067). D4=0.059 and D1=0.091 are appropriately low.

**Did it improve on the best?** Partial recovery from W3's 7.318 (8.284), but still well below W2's all-time best of 9.704. Three consecutive weeks without matching the W2 best.

**What did it teach us?** The GP+GBM ensemble analysis confirmed D3 must be kept very low — W4's D3=0.251 is 6× higher than W2's 0.040, directly explaining the shortfall. D6=0.769 is also far higher than W2's 0.067. The ARD kernel should be learning these constraints but UCB β=2.5 keeps exploring along D3 and D6 because the GP has not ruled out that these high values might be good. They are not. The evidence is now unambiguous: D3 < 0.10 and D6 < 0.15 are hard requirements. The W5 suggestion [0.136, 0.240, 0.025, 0.032, 0.989, 0.204, 0.334, 0.718] has D3=0.025 (correct) but D6=0.204 and D7=0.334, D8=0.718 are all much higher than the W2 best — these deviations may limit the W5 result.

**Strategy for next week (W5):** Submit W5 suggestion as generated. Hard constraints confirmed for W6: D1 < 0.25, D2 < 0.25, D3 < 0.07, D4 < 0.08, D5 > 0.90. D6, D7, D8 are lower-sensitivity (GP ARD and GBM agree) and can be explored more freely, but should be anchored near the W2 best values (D6≈0.07, D7≈0.22, D8≈0.06) until a new best is found.

---

## Surrogate Model Improvement — Output Standardisation (Between W3 and W4)

**Date:** 2026-04-08 · **Applies to:** All functions (F2–F8; F1 already had arcsinh transform)

### What changed

A `"standardize"` Y-transform was added to the GP fitting pipeline in `capstone_app.py` and applied to all functions except F1. Before fitting, the full training set Y values (initial data + portal submissions) are z-scored: `Y_fit = (Y − μ) / σ`. The GP is then fitted on this standardised target instead of raw Y. When displaying the GP mean and std in the dashboard, the transform is reversed so numbers remain interpretable in original units.

The existing `normalize_y=True` setting inside scikit-learn's `GaussianProcessRegressor` was changed to `normalize_y=False` whenever a Y-transform is active, preventing the GP from applying a second z-score on top of the explicit one.

### Why this was necessary

**The acquisition function runs in raw Y units** — the GP's `predict` method undoes its internal normalisation before returning values, so `mean` and `std` come back in whatever scale was passed to `fit`. This meant:

- For F4 (Y range [−32.6, −1.2]), the W1 outlier at −21.254 had a z-score of roughly −3.7 in the raw space. The kernel hyperparameter optimisation was distorted by this single catastrophic point.
- For F5 (Y range [50, 1374]), ξ=0.01 in raw EI meant "improve by 0.01 over a baseline of 1374" — correct in intent (pure exploitation) but only by coincidence of scale, not by principled setting.
- For F7 and F8, β and ξ had implicitly different meanings because the Y ranges (0–2.4 and 5.6–9.7 respectively) differ by 5×.

With explicit standardisation, ξ=0.01 consistently means "require improvement of 0.01 standard deviations above the current best" across all functions. β similarly has a consistent meaning in terms of how many standard deviations of GP uncertainty to add.

### Function-level impact


| Fn  | Raw Y range     | Key benefit of standardise                                                                 |
| --- | --------------- | ------------------------------------------------------------------------------------------ |
| F1  | ≈0 everywhere   | No change — arcsinh already applied                                                        |
| F2  | [−0.07, 0.61]   | Consistent ξ interpretation; minor numerical benefit                                       |
| F3  | [−0.40, −0.018] | ξ=0.02 now means a fixed fraction of σ, not 2% of a −0.4 range                             |
| F4  | [−32.6, −1.2]   | **Primary beneficiary** — outlier at −21.254 no longer distorts kernel fit                 |
| F5  | [50, 1374]      | **Primary beneficiary** — 27× range compressed to unit variance; EI now numerically stable |
| F6  | [−2.57, −0.38]  | Moderate benefit; ξ meaning standardised                                                   |
| F7  | [0.003, 2.36]   | β and ξ now on same scale as other functions                                               |
| F8  | [5.59, 9.70]    | Narrow raw range but positive-only — zero-mean GP prior now correctly centred              |


### What this does not change

- Acquisition function argmax for UCB: UCB is a linear function of mean and std. Scaling both by a constant σ and shifting by μ does not change which candidate point has the highest UCB value. So **UCB suggestions are unaffected in direction**.
- EI suggestions: the shape of the EI surface can change slightly because ξ now means something different in absolute Y units — it is smaller (more exploitation-focused) for functions where σ > 1 and larger for functions where σ < 1. In practice all current functions are in exploit mode so this is benign.
- Visualisation: the GP slice plots and history charts still use raw Y values (the `_prepare_gp` path does not apply the transform), so all displayed numbers remain in the original units shown throughout this document.

---

## Surrogate Model Improvement — Heteroscedastic GP for F2 (Between W4 and W5)

**Date:** 2026-04-08 · **Applies to:** Function 2 only

### Motivation

After W3 we had two observations very close to the known peak region:

- Initial best: `[0.703, 0.927] → 0.611`
- W3 portal: `[0.694, 0.906] → 0.493`

These two points are only 0.023 apart in input space but 0.118 apart in output. The function description confirms it is explicitly noisy. A standard (homoscedastic) GP uses a single constant noise term `alpha = 1e-6` across all training points, so it interprets the 0.118 gap as a genuine spatial gradient — a steep cliff in the landscape. This leads the acquisition function to push away from the initial best toward the "uphill" side, which may be noise rather than structure.

A **heteroscedastic GP** models the noise level as a function of `x`. Points in high-noise regions (like the peak) are assigned a larger `alpha_i`, telling the GP "I expect a larger discrepancy here — do not over-interpret it." The result is wider, more honest uncertainty bands near the peak and a suggestion that properly reflects our genuine uncertainty about where the true maximum sits, rather than chasing a noise-driven gradient.

### What changed

A new `compute_heteroscedastic_alpha(X, Y_fit)` function was added to `capstone_app.py`. It runs immediately after the standardise transform and produces a per-point noise array used as the GP's `alpha` argument.

**Algorithm:**

1. **Leave-one-out (LOO) residuals:** for each of the `n` training points, fit a GP on the remaining `n−1` points and predict the held-out point. The squared prediction error is a local noise estimate. For the peak region, the two nearby observations (0.611 and 0.493) strongly contradict each other during LOO, producing large residuals. In the flat low-Y region, the GP predicts each held-out point well, producing small residuals.
2. **Gaussian kernel smoother** (bandwidth = 0.20 in [0,1] units): smooths each point's noise estimate by taking a weighted average over nearby points. This prevents isolated spikes and ensures the noise map varies smoothly across the input space.
3. **Clip and return** as an array of `alpha` values in z-score units (since `Y_fit` is already standardised). Passed as `alpha=alpha_arr` to `GaussianProcessRegressor` — sklearn supports per-point alpha natively.

The GP is built with `normalize_y=False` (Y is already in z-score units) and the per-point alpha array in the same z-score units. The visualisation path (`_prepare_gp`) also uses heteroscedastic alpha, converting from raw-Y² units to normalised units via `/Y.std()²` so that CI bands in the slice plots also widen at the noisy peak.

### Empirical validation on F2 data (n=14 points)


| Region     | Example point  | Y      | alpha (z-score²) |
| ---------- | -------------- | ------ | ---------------- |
| Noisy peak | [0.703, 0.927] | 0.611  | **1.487**        |
| Noisy peak | [0.694, 0.906] | 0.493  | **1.466**        |
| Flat low-Y | [0.143, 0.349] | −0.066 | 1.182            |
| Flat low-Y | [0.339, 0.214] | −0.014 | 0.936            |


The peak region is assigned ~1.6× more noise than the flat region. The GP will have wider uncertainty bands near [0.70, 0.93] and will not treat the 0.118 gap as a definitive spatial gradient.

### Expected effect on suggestions

Before: the acquisition function could be misled by the apparent downhill gradient from 0.611 → 0.493, generating suggestions that drift away from the true peak.

After: the GP correctly captures that the peak region is uncertain rather than unfavourable. UCB with β=2.5 in this wider-uncertainty region will generate suggestions that sit inside the uncertainty cloud around [0.70, 0.93], rather than being pushed away by a phantom gradient.

### Configuration

`FUNCTION_CONFIG[2]` now includes `"heteroscedastic": True`. A purple `het-GP` badge is displayed in the dashboard alongside the existing `standardize` badge for F2.

---

## Surrogate Comparison — GP vs GBM vs GP+GBM Ensemble for F8 (Between W4 and W5)

**Date:** 2026-04-08 · **Applies to:** Function 8 · **Notebook:** `analysis/04_function8_gpgbm_ensemble.ipynb`

### Motivation

Function 8 is the hardest function: 8 dimensions, 43 observations, and a landscape where only a few dimensions actually matter. Two concerns motivated investigating GBM and a GP+GBM ensemble as alternative surrogates:

1. A pure GP in 8D may struggle to separate genuine structure from noise — it must simultaneously fit the global trend and model local uncertainty across a large input space
2. GBM can capture non-linear interactions explicitly through tree splits, which might be more efficient than the GP's kernel-based interpolation when n is small relative to d

### What was tested

Three surrogates were evaluated in `analysis/04_function8_gpgbm_ensemble.ipynb`:


| Surrogate           | Description                                                                                                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GP (baseline)**   | ARD Matérn 5/2, standardised Y, `normalize_y=False`, UCB β=2.5                                                                                                                |
| **GBM standalone**  | `max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `min_samples_leaf=3`. Bootstrap uncertainty (30 resamples) as a proxy for posterior std                                 |
| **GP+GBM ensemble** | GBM fits the global trend; a residual GP (`C * Matern(ν=2.5)`) fits the remaining variation. Ensemble prediction = GBM mean + residual GP mean; uncertainty = residual GP std |


### Results


| Surrogate             | CV R²             | LOO 95% PI coverage | Predicted Y at suggestion |
| --------------------- | ----------------- | ------------------- | ------------------------- |
| GP (5-fold CV)        | **0.969 ± 0.021** | **0.953** ✓         | 9.89                      |
| GBM (LOO)             | 0.798             | N/A                 | 9.64                      |
| GP+GBM ensemble (LOO) | 0.798             | **0.023** ✗         | 9.65                      |


### What we learned

**1. The GP is far stronger than GBM on this dataset.** GBM's LOO R² of 0.798 versus the GP's 0.969 is a decisive gap. With only 43 points in 8 dimensions, GBM needs many trees (n_estimators=200) to fit the data, which leads to in-sample R² = 0.9995 — essentially memorisation. The LOO score reveals this is not real generalisation.

**2. The ensemble breaks because GBM overfits.** The residual GP receives residuals with near-zero variance (residual variance fraction = 0.001). With nothing to learn, the residual GP collapses to a flat, near-constant posterior with std ≈ 0. The ensemble's LOO 95% PI coverage is 0.023 — catastrophically overconfident — because the residual GP's uncertainty is almost entirely eliminated by the GBM's overfit. If the ensemble were used for acquisition, β=2.5 × std ≈ 0 = no exploration at all.

**3. GBM and GP agree completely on which dimensions matter.** This is the genuinely useful finding from the exercise. Despite their different predictive accuracy, both methods rank the dimensions in the same order:


| Rank | Dim   | GBM permutation importance | GP ARD 1/ℓ |
| ---- | ----- | -------------------------- | ---------- |
| 1    | D3    | 0.386                      | 0.456      |
| 2    | D1    | 0.297                      | 0.317      |
| 3    | D7    | 0.161                      | 0.320      |
| 4    | D2    | 0.081                      | 0.232      |
| 5–8  | D4–D8 | < 0.04                     | < 0.10     |


This cross-validates the ARD kernel's automatic relevance findings independently. D3 and D1 are unambiguously the dominant dimensions. D5, D6, and D8 contribute almost nothing — the GP's ARD assigns them maximum length-scales (ℓ = 10.0), meaning the function barely varies along those axes.

**4. The GP's calibration is excellent.** LOO 95% PI coverage = 0.953 (target = 0.95) is near-perfect. The GP is neither overconfident nor over-conservative, which is exactly what a principled acquisition function requires.

**5. When would GP+GBM be worth trying?** The ensemble would be valuable if the GP LOO R² were poor (say < 0.5) because the landscape had strong non-linearities that the GP's smooth kernel couldn't capture. For F8, the GP is already capturing 97% of variance on held-out points — there is no residual structure left for GBM to find.

### Decision

**Continue with the GP as the sole production surrogate for F8.** No change to `FUNCTION_CONFIG[8]`. The GBM analysis is retained as a validation tool only — confirming D1 and D3 are the dimensions that must stay low.

**W5 strategy implication:** the GP's UCB suggestion explores around D3≈0.23 (higher than the W2 best at D3=0.04), which the GP is uncertain about. A safer option is tight exploitation within the confirmed neighbourhood of the W2 best `[0.21, 0.20, 0.04, 0.04, 0.97, 0.07, 0.22, 0.06]`, perturbing D6 and D7 (both surrogates agree these have low sensitivity and therefore offer the most information gain per query).

---

## Neural Networks and CNNs as BBO Surrogates — Viability Analysis (W4–W5)

**Date:** 2026-04-08 · **Applies to:** All functions · **Notebook:** `analysis/05_nn_surrogate_analysis.ipynb`

### Can a CNN be used for any function?

**No, for all eight functions.** CNNs exploit local spatial correlation in grid-arranged inputs (images, time-series). Their defining inductive bias — a small sliding kernel detecting local patterns — is only useful when neighbouring input dimensions are spatially related.

- **F3–F8:** inputs are unordered hyperparameters, concentrations, or recipe ingredients. Dimension indices have no spatial meaning; shuffling them would not change the function value. CNNs provide zero advantage over a dense MLP and add spatial structure that does not exist.
- **F1–F2:** these are genuinely 2D spatial inputs (position in a [0,1]² field). A CNN *could* in principle exploit spatial smoothness, but the GP's Matérn kernel already models spatial correlation analytically and does so without needing training data to learn the convolution weights. At n=14, a CNN would require a discretised grid with hundreds of cells and corresponding observations. The GP is provably more appropriate at this scale.

### Can a standard MLP be used?

**Not at current sample sizes.** The fundamental problem is the parameter-to-observation ratio:


| Function | n   | MLP(16→8) params | Ratio (params/n) |
| -------- | --- | ---------------- | ---------------- |
| F1–F2    | 14  | 193              | 13.8×            |
| F3       | 19  | 209              | 11.0×            |
| F4–F5    | 24  | 225              | 9.4×             |
| F7       | 33  | 257              | 7.8×             |
| F8       | 43  | 289              | 6.7×             |


Even the smallest two-hidden-layer MLP (16 → 8 units) has 6–14× more free parameters than training observations for every function. An ARD GP for F8 has just **9 hyperparameters** — optimised analytically via marginal-likelihood maximisation, not gradient descent on MSE.

### Empirical comparison — F7 (n=33, 6D) and F8 (n=43, 8D)

LOO cross-validation results from `analysis/05_nn_surrogate_analysis.ipynb`:


| Surrogate           | F7 LOO R² | F8 LOO R² | F8 95% PI coverage |
| ------------------- | --------- | --------- | ------------------ |
| GP (ARD Matérn 5/2) | **0.563** | **0.985** | **0.977**          |
| MLP (16→8)          | −0.091    | 0.887     | N/A                |
| MLP (32→16)         | 0.199     | 0.865     | N/A                |
| Deep Ensemble K=10  | −0.417    | 0.906     | 0.907              |


**F7 is decisive:** all NN variants fail completely. The ensemble LOO R² of −0.417 means it predicts worse than the training mean. F7's output range is [0.003, 2.358] with a skewed distribution, and at n=33 with 6D inputs, none of the MLPs generalise.

**F8 is more nuanced:** the Deep Ensemble reaches R²=0.906, not catastrophically bad. But the GP's 0.985 is still 8 percentage points higher and its uncertainty calibration (0.977 vs target 0.950) is near-perfect, compared to the ensemble's 0.907. Even the best-case NN is clearly outperformed.

### Why does the GP win at this data scale?

1. **Parameter efficiency.** 9 GP hyperparameters vs 289 MLP parameters for F8. The GP achieves higher accuracy with dramatically fewer degrees of freedom.
2. **Marginal-likelihood optimisation.** GP hyperparameters are fitted by maximising the log-marginal-likelihood, which naturally balances fit quality against model complexity — a built-in form of regularisation. MLP gradient descent on MSE has no such analytical safeguard.
3. **Principled uncertainty.** GP posterior variance is derived from first principles and has guaranteed properties. Deep Ensemble uncertainty is an empirical spread across K independently trained models — it underestimates uncertainty in sparsely sampled regions because all K models extrapolate the same learned function confidently.

### Learning curve — when would a NN become viable?

The learning curves (Section 6 of the notebook) show that a GP consistently outperforms the MLP across all tested training sizes (n=8 to 33). A genuine crossover would require approximately **n ≥ 100–200 observations in 8D** for a properly regularised MLP — roughly 3–5× the current F8 dataset size.

### Most promising future direction: Deep Kernel Learning

If sample sizes grow beyond ~60 observations for any function, the most promising NN-based approach would be **Deep Kernel Learning (DKL)**: train a small NN as a feature extractor (ℝ^d → ℝ^k, where k < d), then fit a GP on those learned features. The NN learns a problem-specific kernel; the GP retains calibrated uncertainty. For F8 this could learn a 3–4D feature space separating the high-Y peak (D1 and D3 low, D5 high) from the rest. Requires GPyTorch (not currently available).

### Decision

**No change to any surrogate configuration.** The GP remains the production surrogate for all eight functions. The NN analysis confirms the GP choice is correct, not conservative.

---

## Week 5 — 2026-04-10

### Function 1 — 2D Contamination Field

**Submitted:** [0.080000, 0.480000] → **Y ≈ 0.000**

**Acquisition:** UCB β=2.0, Matérn 5/2, arcsinh Y-transform

**Exploration or exploitation?** Manual probe — hand-specified per W4 strategy to target the lower-left quadrant for the first time.

**Did it improve on the best?** No. Six portal submissions across five quadrant regions, all effectively zero. No signal has been detected anywhere in the [0,1]² domain.

**What did it teach us?** The lower-left (X₁=0.08, X₂=0.48) also returns zero, completing the four-quadrant sweep: top-right (W1), lower-right (W2), left-mid (W3), top-right again (W4, accidental revisit), lower-left (W5). Every major region has been tested at the quadrant level. The contamination hotspot either sits in an extremely narrow sub-region not yet hit, or requires a precision far below quadrant-level targeting. With the GP having never seen a non-zero output, acquisition function guidance is essentially random — it cannot learn a landscape it has never detected. GP guidance is futile until a non-zero reading appears.

**Strategy for next week (W6):** Dense sub-quadrant probe. Submit near [0.12, 0.52] — a short step from W5 within the lower-left, probing whether the hotspot is near the W5 point rather than at an extreme corner. If still zero, the remaining weeks should run a geometric sweep: diagonals [0.25, 0.25], [0.50, 0.15], [0.15, 0.50] to maximise domain coverage by distance from all prior queries.

---

### Function 2 — 2D Noisy Log-Likelihood

**Submitted:** [0.697709, 0.942183] → **Y = 0.5130**

**Acquisition:** UCB β=2.5, Matérn 5/2, heteroscedastic GP, standardised Y

**Exploration or exploitation?** Tight exploitation — near-identical X₁ to the all-time best (0.698 vs 0.700), X₂ pulled back slightly from W4 (0.942 vs 0.961).

**Did it improve on the best?** No — regression. W4 achieved 0.6485; W5 returned 0.513 (−20.9%). The all-time best remains W4: 0.6485 at [0.699929, 0.961372].

**What did it teach us?** The only meaningful coordinate difference between W4 and W5 is X₂: 0.961 (W4, best ever) vs 0.942 (W5, worse). X₁ ≈ 0.700 was held constant in both. The 0.136 output gap produced by a 0.019 step in X₂ indicates the function is steeply peaked along the X₂ axis in this region. The optimal X₂ is close to 0.96 — reducing to 0.94 already costs 20% of output. The noisy function and heteroscedastic GP are correctly managing the uncertainty, but the X₂ = 0.96 band is now well-established as the peak location.

**Strategy for next week (W6):** Restore X₂ to the W4 level and probe slightly higher. Submit near [0.699, 0.963] — a +0.002 step in X₂ from W4. Hard constraint: X₂ ≥ 0.955 in all future submissions. Keep UCB β=2.5.

---

### Function 3 — 3D Drug Compounds

**Submitted:** [0.438733, 0.461281, 0.502799] → **Y = −0.008967**

**Acquisition:** EI β=1.96, ξ=0.02, Matérn 5/2, standardised Y

**Exploration or exploitation?** Tight exploitation — returned directly to the W2 best neighbourhood as planned, with a slightly different position in Compound B.

**Did it improve on the best?** **Yes — new all-time best.** Improved from −0.0182 (W2) to −0.00897, a 50.7% further reduction in adverse reaction score. Cumulative improvement from the initial best (−0.0348): 74.2%.

**What did it teach us?** W5 coordinates vs W2 best: A=0.439 (≈W2's 0.446), B=0.461 (+36% above W2's 0.339), C=0.503 (≈W2's 0.486). The improvement came primarily from a higher Compound B — 0.461 vs 0.339. Despite previous evidence suggesting the best B was mid-low (0.30–0.35), a mid-range B≈0.46 produced a better result. The optimal B is likely in [0.40–0.50] rather than [0.30–0.35]. The EI with ξ=0.02 explored gently within the W2 neighbourhood and found genuine improvement — the right level of ξ for this stage.

**Strategy for next week (W6):** Reduce ξ to 0.01, tighten around the W5 best. Target near [0.44, 0.50, 0.47] — a small nudge in B toward 0.50. Soft constraints: A ∈ [0.40, 0.55], B ∈ [0.42, 0.55], C ∈ [0.46, 0.55]. Do not deviate more than 0.05 per dimension from W5 coordinates.

---

### Function 4 — 4D Warehouse Hyperparameters

**Submitted:** [0.464534, 0.420432, 0.270045, 0.389012] → **Y = −1.8974**

**Acquisition:** UCB β=2.0, ξ=0.05, Matérn 5/2, standardised Y

**Exploration or exploitation?** Exploitation — mid-range values, P3 locked at 0.270 as planned.

**Did it improve on the best?** No — fourth consecutive week without matching the W2 best. Partial recovery from W4: −2.370 → −1.897, but still 61% below the W2 best of −1.177.

**What did it teach us?** The P3 sweep across four weeks is now instructive: P3=0.311 → Y=−1.177 (W2, best); P3=0.385 → Y=−1.568 (W3); P3=0.499 → Y=−2.370 (W4); P3=0.270 → Y=−1.897 (W5). This is a bowl-shaped response with a minimum around P3≈0.31. Moving P3 either higher or lower from 0.311 degrades performance, but asymmetrically — P3 too high (0.499) is worse than P3 too low (0.270). The other three parameters were essentially at W2 values in W5 (P1=0.465 vs 0.460, P2=0.420 vs 0.413, P4=0.389 vs 0.405), confirming P3 alone is responsible for the gap. The W2 value of P3=0.311 was correct from the start.

**Strategy for next week (W6):** Lock P3 at exactly 0.311 (or within ±0.010). Submit near [0.462, 0.414, 0.311, 0.404] — reproducing the W2 best as precisely as possible. Switch to EI ξ=0.01. If this reproduces W2's −1.177 or better, the next step is micro-perturbation of P1/P2/P4 one at a time.

---

### Function 5 — 4D Chemical Yield

**Submitted:** [0.338929, 0.838032, 0.945674, 0.872000] → **Y = 1412.628**

**Acquisition:** EI β=1.5, ξ=0.01, RBF kernel, standardised Y

**Exploration or exploitation?** Tight exploitation — EI ξ=0.01 self-corrected from the W4 coordinate drift.

**Did it improve on the best?** **Yes — new all-time best.** W4 had regressed to 1124.92; W5 surpassed the previous all-time best (W3: 1374.52) to reach 1412.63 (+2.8%). Four of five portal submissions have now been in the top-three range: the one exception was W4's "mean" acquisition overshoot.

**What did it teach us?** Switching back from "mean" to EI ξ=0.01 immediately self-corrected. The critical coordinate shifts: C1 reduced from 0.415 (W4) back to 0.339, and C4 restored from 0.797 (W4) back to 0.872. Both of these returned the coordinates to within the confirmed hard-constraint bounds (C1 ∈ [0.28, 0.42], C4 > 0.85). The unimodal landscape means small but correct adjustments reliably find improvements. The peak region is now well-bracketed: C≈[0.33–0.38, 0.838, 0.945, 0.872].

**Strategy for next week (W6):** Continue EI ξ=0.01. Target near [0.320, 0.840, 0.955, 0.875] — a careful step from W5 with a slight C3 increase (0.946 → 0.955) and C2 nudge (0.838 → 0.840). C1 is narrowing toward [0.30–0.35]; do not push above 0.38.

---

### Function 6 — 5D Cake Recipe

**Submitted:** [0.342811, 0.522702, 0.603234, 0.750991, 0.141019] → **Y = −0.341297**

**Acquisition:** Mean (pure exploitation), Matérn 5/2, standardised Y

**Exploration or exploitation?** Pure exploitation — GP posterior mean maximiser, anchored on the W3 best neighbourhood.

**Did it improve on the best?** **Yes — new all-time best.** W3 had been the previous best at −0.384; W5 reached −0.341 (+11.2%). Two consecutive improvements after the W4 regression: W3=−0.384 → W4=−1.294 (collapse) → W5=−0.341. Now 52.2% better than the initial best (−0.714).

**What did it teach us?** The W5 recipe (Flour=0.343, Sugar=0.523, Eggs=0.603, Butter=0.751, Milk=0.141) revises the earlier intuition that Sugar must be kept low. Sugar=0.523 is the highest ever used in a top-performing submission, yet it produced a new best — evidently Sugar is not a primary driver of penalty in this range. Butter=0.751 and Eggs=0.603 are the reliable anchors: Butter in [0.70, 0.78] and Eggs above 0.50 have appeared in every good result. Milk at 0.141 remains low and beneficial. Switching from EI to pure "mean" acquisition successfully prevented the kind of large exploratory jump that caused W4's regression.

**Strategy for next week (W6):** Continue "mean" acquisition. Target near [0.35, 0.52, 0.62, 0.75, 0.14] — small Eggs increase (+0.017). Butter and Milk are stable; Sugar can remain in [0.45, 0.55]. Hard constraints remain: Butter ∈ [0.70, 0.78], Eggs > 0.50, Milk < 0.20.

---

### Function 7 — 6D GBM Hyperparameters

**Submitted:** [0.095050, 0.364801, 0.337000, 0.317000, 0.362000, 0.721151] → **Y = 2.3562**

**Acquisition:** EI β=1.96, ξ=0.05, Matérn 5/2, ARD, standardised Y

**Exploration or exploitation?** Pure exploitation — coordinates virtually identical to the W2 all-time best.

**Did it improve on the best?** Essentially matched. Y=2.3562 vs W2's 2.3576 (gap of 0.0014). Two weeks of severe regression (W3: 1.931, W4: 0.745) were fully reversed by returning to the W2 coordinates.

**What did it teach us?** The near-perfect reproduction of the W2 result with near-identical inputs is highly significant: **Function 7 is effectively deterministic**. Same input coordinates → same output, within 0.001. This has two implications: (1) the W2/W5 value of 2.357 is a reliable, noise-free estimate — it is not a lucky outlier; (2) beating this best requires genuinely better coordinates, not lucky re-sampling. The landscape is sharply peaked: a 0.06 deviation in dim1 (W3) cost 18%; a 0.26 deviation in dim2 (W4) cost 68%. The peak is narrow in dim1 (n_estimators ≈ 0.095) and dim2 (learning_rate ≈ 0.365). Regularisation dim6=0.721 appears stable.

**Strategy for next week (W6):** Single-dimension micro-perturbation. Proposed: reduce dim1 only (0.095 → 0.082), hold all others at W2/W5 values. If Y < 2.355, conclude dim1=0.095 is essentially at the optimum; pivot to testing dim2 (+0.005). Do not deviate more than one dimension at a time, and never more than ±0.015 per dimension from [0.095, 0.365, 0.337, 0.317, 0.362, 0.721].

---

### Function 8 — 8D ML Hyperparameters

**Submitted:** [0.136428, 0.239528, 0.025387, 0.032045, 0.988952, 0.204302, 0.333825, 0.718319] → **Y = 9.8001**

**Acquisition:** UCB β=2.5, ξ=0.1, Matérn 5/2, ARD, standardised Y

**Exploration or exploitation?** Exploitation of the critical dimensions (D1, D3, D4, D5) combined with exploration in the low-sensitivity dimensions (D6, D7, D8).

**Did it improve on the best?** **Yes — new all-time best.** Improved from W2's 9.7035 to 9.8001 (+1.0%). Three weeks of regression (W3: 7.318, W4: 8.284) reversed. The GP+GBM ensemble analysis prediction was validated.

**What did it teach us?** W5 directly confirmed three findings from the between-W4/W5 analysis: (1) D3=0.025 (well below the 0.07 hard limit) and D5=0.989 (above 0.90) are necessary and sufficient for high output; (2) D6=0.204, D7=0.334, D8=0.718 — all substantially higher than W2's 0.067/0.222/0.061 — did not prevent a new best, validating GP ARD's assignment of maximum length-scales to these dimensions; (3) D1=0.136 and D2=0.240 are acceptable (both below 0.25), with D1 now confirmed in the working range. The GP correctly explored D6–D8 without penalty, freeing future queries to focus on the critical D1/D3/D4/D5 subspace. Notably, the W4 reflection's concern that D6=0.204 "may limit the W5 result" was proven wrong — low sensitivity in D6/D7/D8 is confirmed beyond doubt.

**Strategy for next week (W6):** Hard constraints: D1 < 0.18, D3 < 0.04, D4 < 0.05, D5 > 0.97. Push D1 lower (0.136 → 0.05–0.10 range) to test whether a further reduction beyond W5 improves output. D2 can remain near 0.20–0.25. D6/D7/D8 can be varied freely — the GP should explore along these axes at will.

---

*Reflections for Week 6 and beyond will be appended below.*