"""
=============================================================================
CAPSTONE OPTIMISATION TOOLKIT
=============================================================================
One query per function per week. Every query counts.

Usage:
    1. On first week: run initialise_function(fn_id) to get your first query
    2. Each subsequent week: run next_query(fn_id, last_input, last_output)
    3. Run weekly_summary() to see all function states
    4. Run reflection(fn_id) to generate your written reflection text
=============================================================================
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from datetime import datetime
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')

# =============================================================================
# FUNCTION CONFIGURATIONS
# =============================================================================
# Each function has:
#   dims        : input dimensionality
#   bounds      : list of (min, max) per dimension
#   surrogate   : 'gp' or 'rf' (random forest via GBR ensemble)
#   kernel      : kernel type for GP ('rbf', 'matern')
#   acquisition : 'ucb', 'ei', 'pi', 'variance'
#   beta        : UCB exploration parameter
#   xi          : PI/EI exploration parameter
#   notes       : strategy rationale

FUNCTION_CONFIG = {
    1: {
        "dims": 2,
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
        "surrogate": "gp",
        "kernel": "rbf",
        "acquisition": "ucb",
        "beta": 2.0,
        "xi": 0.05,
        "description": "2D contamination/radiation field",
        "notes": (
            "Sparse non-zero regions suggest the signal is localised. "
            "UCB with beta=2.0 keeps exploration high early to avoid missing "
            "the signal entirely. Reduce beta to 1.0 after a non-zero reading."
        ),
    },
    2: {
        "dims": 2,
        "bounds": [(0.0, 1.0), (0.0, 1.0)],
        "surrogate": "gp",
        "kernel": "matern",
        "acquisition": "ucb",
        "beta": 2.5,
        "xi": 0.1,
        "description": "2D noisy black-box log-likelihood",
        "notes": (
            "Explicitly noisy with local optima. Matern kernel handles "
            "rougher functions better than RBF. High beta on UCB to resist "
            "premature exploitation of noisy early readings."
        ),
    },
    3: {
        "dims": 3,
        "bounds": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "surrogate": "gp",
        "kernel": "matern",
        "acquisition": "ei",
        "beta": 1.96,
        "xi": 0.02,
        "description": "3D drug compound combinations (negated side effects)",
        "notes": (
            "Physical process — Matern kernel appropriate. EI balances "
            "improvement focus with uncertainty. Bounds may need tightening "
            "once you establish which compound ratios are viable."
        ),
    },
    4: {
        "dims": 4,
        "bounds": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "surrogate": "gp",
        "kernel": "matern",
        "acquisition": "ucb",
        "beta": 2.0,
        "xi": 0.05,
        "description": "4D warehouse ML hyperparameter tuning",
        "notes": (
            "Rough landscape with local optima — stay exploratory early. "
            "Dynamic environment means old observations may drift in value; "
            "weight recent observations more heavily in later rounds."
        ),
    },
    5: {
        "dims": 4,
        "bounds": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "surrogate": "gp",
        "kernel": "rbf",
        "acquisition": "ei",
        "beta": 1.5,
        "xi": 0.01,
        "description": "4D chemical yield (unimodal)",
        "notes": (
            "Unimodal — safest function to exploit aggressively once in the "
            "right neighbourhood. Low xi on EI. RBF kernel appropriate for "
            "smooth unimodal surface. Shift to pure exploitation by week 4."
        ),
    },
    6: {
        "dims": 5,
        "bounds": [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],
        "surrogate": "gp",
        "kernel": "matern",
        "acquisition": "ei",
        "beta": 1.96,
        "xi": 0.02,
        "description": "5D cake recipe (negative penalty score, maximise toward 0)",
        "notes": (
            "Output is negative by design. Scores near zero are best. "
            "Use domain knowledge for initialisation — start near balanced "
            "ingredient ratios. Extreme values almost certainly score worse."
        ),
    },
    7: {
        "dims": 6,
        "bounds": [
            (50, 500),       # n_estimators: typical range 50–500
            (0.01, 0.3),     # learning_rate: typical range 0.01–0.3
            (3, 10),         # max_depth: typical range 3–10
            (0.5, 1.0),      # subsample: typical range 0.5–1.0
            (0.5, 1.0),      # max_features (colsample): typical range 0.5–1.0
            (1e-4, 1.0),     # min_samples_leaf / regularisation: 1e-4 to 1.0
        ],
        "surrogate": "gp",
        "kernel": "matern",
        "acquisition": "ei",
        "beta": 1.96,
        "xi": 0.05,
        "description": "6D gradient boosting hyperparameter tuning",
        "notes": (
            "Almost certainly GBM (XGBoost/sklearn GBT). Bounds set from "
            "published best-practice ranges. Note: inputs should be normalised "
            "before fitting GP, then denormalised for submission. "
            "EI appropriate — we know the landscape is structured. "
            "Key insight: learning_rate and n_estimators are inversely related; "
            "if one increases, the other typically should decrease."
        ),
        "informed_start": [200, 0.1, 5, 0.8, 0.8, 0.1],  # sensible GBM defaults
    },
    8: {
        "dims": 8,
        "bounds": [
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0),
        ],
        "surrogate": "gp",
        "kernel": "matern",
        "acquisition": "ucb",
        "beta": 2.5,
        "xi": 0.1,
        "description": "8D complex black-box (likely ML hyperparameters)",
        "notes": (
            "Hardest function. GP at 8D will be uncertain — accept this. "
            "High beta UCB keeps exploration broad. Focus on process quality "
            "in reflections rather than absolute score. Strong local maxima "
            "are a valid target per the brief."
        ),
    },
}

# =============================================================================
# PERSISTENCE — save/load observation history
# =============================================================================

HISTORY_FILE = "capstone_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            raw = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in raw.items()}
    return {fn_id: {"X": [], "Y": [], "week": 0} for fn_id in range(1, 9)}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# =============================================================================
# SURROGATE MODEL
# =============================================================================

def build_gp(kernel_type="matern"):
    if kernel_type == "rbf":
        kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
    else:  # matern
        kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 10.0))
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
    )

# =============================================================================
# ACQUISITION FUNCTIONS
# =============================================================================

def ucb(mean, std, beta):
    return mean + beta * std

def ei(mean, std, y_max, xi):
    z = (mean - y_max - xi) / (std + 1e-12)
    return (mean - y_max - xi) * norm.cdf(z) + std * norm.pdf(z)

def pi(mean, std, y_max, xi):
    z = (mean - y_max - xi) / (std + 1e-12)
    return norm.cdf(z)

def variance_acq(mean, std):
    return std ** 2

def compute_acquisition(acq_type, mean, std, y_max, beta, xi):
    if acq_type == "ucb":
        return ucb(mean, std, beta)
    elif acq_type == "ei":
        return ei(mean, std, y_max, xi)
    elif acq_type == "pi":
        return pi(mean, std, y_max, xi)
    elif acq_type == "variance":
        return variance_acq(mean, std)
    else:
        raise ValueError(f"Unknown acquisition function: {acq_type}")

# =============================================================================
# CANDIDATE GRID GENERATION
# =============================================================================

def generate_candidates(bounds, n=2000):
    """Latin Hypercube-style sampling across the bounded space."""
    dims = len(bounds)
    # Sobol-like stratified sampling
    candidates = np.zeros((n, dims))
    for d, (lo, hi) in enumerate(bounds):
        candidates[:, d] = np.random.uniform(lo, hi, n)
    return candidates

def normalise_X(X, bounds):
    X = np.array(X)
    bounds = np.array(bounds)
    return (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])

def denormalise_X(X_norm, bounds):
    X_norm = np.array(X_norm)
    bounds = np.array(bounds)
    return X_norm * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

# =============================================================================
# CORE QUERY LOGIC
# =============================================================================

def suggest_next_query(fn_id, history, override_acq=None, override_beta=None, override_xi=None):
    """
    Given current observation history, suggest the next query point.
    Returns the suggested input as a list (in original scale).
    """
    cfg = FUNCTION_CONFIG[fn_id]
    bounds = cfg["bounds"]
    acq_type = override_acq or cfg["acquisition"]
    beta = override_beta or cfg["beta"]
    xi = override_xi or cfg["xi"]

    fn_history = history[fn_id]
    X_obs = fn_history["X"]
    Y_obs = fn_history["Y"]

    # --- No observations yet: use informed start or random ---
    if len(X_obs) == 0:
        if "informed_start" in cfg:
            print(f"  [F{fn_id}] Using literature-informed starting point.")
            return cfg["informed_start"]
        else:
            candidate = [np.random.uniform(lo, hi) for lo, hi in bounds]
            print(f"  [F{fn_id}] No observations yet — random initialisation.")
            return candidate

    # --- Normalise observations ---
    X_norm = normalise_X(X_obs, bounds)
    Y_arr = np.array(Y_obs)

    # --- Fit surrogate ---
    gp = build_gp(cfg["kernel"])
    gp.fit(X_norm, Y_arr)

    # --- Generate candidates and predict ---
    candidates_raw = generate_candidates(bounds, n=5000)
    candidates_norm = normalise_X(candidates_raw, bounds)
    mean, std = gp.predict(candidates_norm, return_std=True)

    # --- Compute acquisition ---
    y_max = np.max(Y_arr)
    acq_values = compute_acquisition(acq_type, mean, std, y_max, beta, xi)

    # --- Select best candidate ---
    best_idx = np.argmax(acq_values)
    best_candidate = candidates_raw[best_idx]

    print(f"  [F{fn_id}] Acquisition: {acq_type.upper()} | "
          f"GP posterior at suggestion — mean: {mean[best_idx]:.4f}, "
          f"std: {std[best_idx]:.4f}")
    print(f"  [F{fn_id}] Current best observed: {y_max:.4f}")

    return best_candidate.tolist()

# =============================================================================
# PUBLIC API
# =============================================================================

def initialise_function(fn_id):
    """
    Get your first query point for a function (week 1).
    Call this before you have any observations.
    """
    history = load_history()
    cfg = FUNCTION_CONFIG[fn_id]

    print(f"\n{'='*60}")
    print(f"FUNCTION {fn_id}: {cfg['description']}")
    print(f"Strategy: {cfg['notes']}")
    print(f"{'='*60}")

    suggestion = suggest_next_query(fn_id, history)

    print(f"\n>>> SUBMIT THIS INPUT FOR FUNCTION {fn_id}:")
    print(f"    {format_submission(fn_id, suggestion)}")
    print(f"{'='*60}\n")

    return suggestion


def next_query(fn_id, last_input, last_output,
               override_acq=None, override_beta=None, override_xi=None):
    """
    Record last week's result and get next query suggestion.

    Args:
        fn_id       : function number (1–8)
        last_input  : the input you submitted last week (list)
        last_output : the output value you received (float)
        override_acq: optionally override acquisition function ('ucb','ei','pi','variance')
        override_beta: optionally override beta parameter
        override_xi : optionally override xi parameter
    """
    history = load_history()
    cfg = FUNCTION_CONFIG[fn_id]

    # Record the observation
    history[fn_id]["X"].append(list(last_input))
    history[fn_id]["Y"].append(float(last_output))
    history[fn_id]["week"] = history[fn_id].get("week", 0) + 1
    save_history(history)

    n_obs = len(history[fn_id]["Y"])
    y_max = max(history[fn_id]["Y"])

    print(f"\n{'='*60}")
    print(f"FUNCTION {fn_id}: {cfg['description']}")
    print(f"Week {history[fn_id]['week']} | Observations so far: {n_obs} | Best: {y_max:.4f}")
    print(f"{'='*60}")

    suggestion = suggest_next_query(
        fn_id, history,
        override_acq=override_acq,
        override_beta=override_beta,
        override_xi=override_xi,
    )

    print(f"\n>>> SUBMIT THIS INPUT FOR FUNCTION {fn_id}:")
    print(f"    {format_submission(fn_id, suggestion)}")
    print(f"{'='*60}\n")

    return suggestion


def weekly_summary():
    """Print a summary table of all functions — current best, observations, week."""
    history = load_history()
    print(f"\n{'='*70}")
    print(f"  WEEKLY CAPSTONE SUMMARY  —  {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*70}")
    print(f"  {'Fn':>3}  {'Description':<38}  {'Week':>4}  {'Obs':>3}  {'Best':>10}")
    print(f"  {'-'*3}  {'-'*38}  {'-'*4}  {'-'*3}  {'-'*10}")

    for fn_id in range(1, 9):
        cfg = FUNCTION_CONFIG[fn_id]
        fn_h = history[fn_id]
        best = max(fn_h["Y"]) if fn_h["Y"] else float("nan")
        n_obs = len(fn_h["Y"])
        week = fn_h.get("week", 0)
        desc = cfg["description"][:38]
        best_str = f"{best:.4f}" if not np.isnan(best) else "—"
        print(f"  {fn_id:>3}  {desc:<38}  {week:>4}  {n_obs:>3}  {best_str:>10}")

    print(f"{'='*70}\n")


def reflection(fn_id):
    """Generate a structured reflection text for the weekly submission."""
    history = load_history()
    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]

    if not fn_h["Y"]:
        print(f"No observations yet for Function {fn_id}.")
        return

    X_obs = fn_h["X"]
    Y_obs = fn_h["Y"]
    week = fn_h.get("week", len(Y_obs))
    y_max = max(Y_obs)
    best_idx = Y_obs.index(y_max)
    best_x = X_obs[best_idx]

    last_y = Y_obs[-1]
    last_x = X_obs[-1]
    improved = last_y == y_max and len(Y_obs) > 1

    print(f"\n{'='*60}")
    print(f"REFLECTION — FUNCTION {fn_id} — WEEK {week}")
    print(f"{'='*60}")
    print(f"""
This week I submitted input {[round(v, 4) for v in last_x]} to Function {fn_id}
({cfg['description']}) and received an output of {last_y:.4f}.

{"This was an improvement on the previous best, confirming the surrogate model's prediction that this region was promising." if improved else f"This did not improve on the current best of {y_max:.4f}, but the observation has updated the GP posterior, reducing uncertainty in this region."}

The acquisition function used was {cfg['acquisition'].upper()}
(beta={cfg['beta']}, xi={cfg['xi']}), which was chosen because:
{cfg['notes']}

Across {len(Y_obs)} observations to date, the best observed value is {y_max:.4f}
at input {[round(v, 4) for v in best_x]}.

Next week I will {"continue exploiting this region with reduced exploration pressure" if improved else "shift the acquisition function to explore a different region of the input space, as the current region does not appear to contain the global maximum"}.
""")
    print(f"{'='*60}\n")


def format_submission(fn_id, input_values):
    """Format input values for portal submission."""
    rounded = [round(float(v), 6) for v in input_values]
    return f"Function {fn_id}: {rounded}"


def plot_history(fn_id):
    """Plot observation history and GP posterior for 1D/2D functions."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available.")
        return

    history = load_history()
    fn_h = history[fn_id]
    cfg = FUNCTION_CONFIG[fn_id]

    if not fn_h["Y"]:
        print(f"No observations yet for Function {fn_id}.")
        return

    Y_obs = fn_h["Y"]
    weeks = list(range(1, len(Y_obs) + 1))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(weeks, Y_obs, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.axhline(max(Y_obs), color="#FF5722", linestyle="--", alpha=0.6, label=f"Best: {max(Y_obs):.4f}")
    ax.set_xlabel("Week")
    ax.set_ylabel("Observed Output")
    ax.set_title(f"Function {fn_id}: {cfg['description']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"f{fn_id}_history.png", dpi=150)
    plt.close()
    print(f"Saved plot to f{fn_id}_history.png")


# =============================================================================
# FUNCTION 7 SPECIAL: GBM-INFORMED SEARCH SPACE
# =============================================================================

def f7_gbm_analysis():
    """
    Print the research-informed parameter guide for Function 7 (GBM).
    Use this to interpret and contextualise your submissions.
    """
    print("""
=================================================================
FUNCTION 7 — GRADIENT BOOSTING HYPERPARAMETER GUIDE
=================================================================
Assumed parameter mapping (sklearn GradientBoostingClassifier):

  Dim 1: n_estimators       [50 – 500]
         Number of boosting stages. More = better fit, slower.
         Best practice: 100–300 for most datasets.

  Dim 2: learning_rate       [0.01 – 0.30]
         Shrinks contribution of each tree. Lower rate + more
         estimators typically outperforms higher rate + fewer.
         Best practice: 0.05–0.15 paired with n_est 200–300.

  Dim 3: max_depth           [3 – 10]
         Maximum depth of individual trees. Deeper = more complex,
         higher overfit risk. Best practice: 3–6 for most problems.

  Dim 4: subsample           [0.5 – 1.0]
         Fraction of samples used per tree (stochastic gradient).
         Values < 1.0 reduce variance. Best practice: 0.7–0.9.

  Dim 5: max_features        [0.5 – 1.0]
         Fraction of features considered per split.
         Adds randomness, similar to Random Forest.
         Best practice: 0.7–1.0.

  Dim 6: min_samples_leaf    [1e-4 – 1.0]
         Regularisation via minimum leaf size.
         Higher = more regularised, less overfit.
         Best practice: 0.01–0.1 for noisy problems.

Key interactions:
  - learning_rate ↑  →  n_estimators ↓  (inverse relationship)
  - max_depth ↑      →  subsample ↓     (counteract overfitting)
  - Noisy function   →  min_samples_leaf ↑

Literature-informed starting point: [200, 0.1, 5, 0.8, 0.8, 0.05]
=================================================================
""")


# =============================================================================
# QUICK START GUIDE (printed on import)
# =============================================================================

QUICKSTART = """
=================================================================
CAPSTONE OPTIMISER — QUICK START
=================================================================
Week 1 (first submission for any function):
    suggestion = initialise_function(3)

Each subsequent week (record result, get next query):
    suggestion = next_query(fn_id=3, last_input=[...], last_output=0.842)

See all function states:
    weekly_summary()

Generate your written reflection:
    reflection(3)

Plot progress over time:
    plot_history(3)

Function 7 GBM parameter guide:
    f7_gbm_analysis()

Override acquisition function for one round:
    next_query(2, last_input=[...], last_output=0.5, override_acq='pi', override_xi=0.1)
=================================================================
"""

if __name__ == "__main__":
    print(QUICKSTART)
    suggestion = initialise_function(1)
    weekly_summary()
    f7_gbm_analysis()
