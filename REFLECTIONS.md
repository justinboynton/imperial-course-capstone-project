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

*Reflections for subsequent weeks will be appended below.*
