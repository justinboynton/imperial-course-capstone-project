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

## Lessons Learned

| Week | Lesson |
|------|--------|
| 1 | **Always integrate all available data before generating suggestions.** The initial .npy data is the most important training signal; ignoring it wastes queries. |
| 1 | **Meta-tracking matters.** Without recording acq/β/ξ/kernel per submission, it is impossible to review why a particular query was made. Now tracked automatically. |
| 1 | **Kernel choice is a one-time structural decision.** RBF was wrong for F1 — changed to Matérn after analysis. Not appropriate to oscillate between kernels. |
| 1 | **Over-exploration in week 1 has downstream cost.** With ~8 remaining queries per function, any query below the initial best is a wasted opportunity. |
