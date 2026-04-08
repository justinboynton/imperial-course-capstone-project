"""
=============================================================================
CAPSTONE OPTIMISATION — STREAMLIT DASHBOARD
=============================================================================
Run with:  streamlit run capstone_app.py
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
import os
import math
import warnings
from datetime import datetime
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:
    import anthropic as _anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

warnings.filterwarnings("ignore")

ANTHROPIC_MODEL = "claude-sonnet-4-6"

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Capstone Optimiser",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# STYLING
# =============================================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600&family=Syne:wght@400;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
  }

  .stApp {
    background: #0a0e1a;
    background-image:
      linear-gradient(rgba(0,212,255,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,212,255,0.025) 1px, transparent 1px);
    background-size: 40px 40px;
  }

  /* Hide default streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

  /* Custom card */
  .fn-card {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.75rem;
    position: relative;
    transition: border-color 0.2s;
    cursor: pointer;
  }
  .fn-card:hover { border-color: #00d4ff44; }
  .fn-card-active { border-color: #00d4ff !important; background: #1a2235 !important; }
  .fn-card-bar {
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 12px 12px 0 0;
  }

  .fn-number {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 7px;
    font-weight: 700; font-size: 0.85rem; color: #000;
    margin-right: 0.6rem; vertical-align: middle;
  }

  .fn-title { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
  .fn-dims  { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #64748b; }

  .stat-val { font-family: 'JetBrains Mono', monospace; font-size: 1rem; font-weight: 700; }
  .stat-lbl { font-size: 0.62rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }

  .pill {
    display: inline-block; padding: 0.15rem 0.5rem; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 600;
    margin-right: 0.3rem;
  }
  .pill-ucb  { background: rgba(0,212,255,0.15); color: #00d4ff; }
  .pill-ei   { background: rgba(16,185,129,0.15); color: #10b981; }
  .pill-pi   { background: rgba(124,58,237,0.15); color: #a78bfa; }
  .pill-var  { background: rgba(245,158,11,0.15); color: #f59e0b; }
  .pill-mean { background: rgba(239,68,68,0.15);  color: #f87171; }

  .section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #64748b; margin-bottom: 0.75rem; padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2d45;
  }

  .suggestion-box {
    background: #0a0e1a; border: 1px solid #00d4ff;
    border-radius: 8px; padding: 0.9rem 1rem; margin-top: 0.75rem;
  }
  .suggestion-label { font-size: 0.62rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.3rem; font-family: 'JetBrains Mono', monospace; }
  .suggestion-value { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: #00d4ff; font-weight: 600; word-break: break-all; }

  .strategy-box {
    background: #0a0e1a; border: 1px solid #1e2d45;
    border-radius: 8px; padding: 0.9rem; font-size: 0.78rem;
    line-height: 1.65; color: #94a3b8;
  }

  .reflection-box {
    background: #0a0e1a; border: 1px solid #1e2d45;
    border-radius: 8px; padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; line-height: 1.7; color: #e2e8f0;
  }

  .obs-row {
    display: grid; grid-template-columns: 36px 1fr auto auto;
    gap: 0.5rem; padding: 0.3rem 0;
    border-bottom: 1px solid #1e2d4530;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    align-items: center;
  }
  .obs-week { color: #64748b; }
  .obs-input { color: #cbd5e1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .obs-output { color: #10b981; font-weight: 600; text-align: right; }
  .obs-best { color: #f59e0b !important; }
  .obs-meta { color: #64748b; font-size: 0.62rem; text-align: right; white-space: nowrap; }

  .ai-box {
    background: #0a0e1a; border: 1px solid #7c3aed44;
    border-radius: 8px; padding: 1.1rem 1.2rem;
    font-size: 0.82rem; line-height: 1.75; color: #cbd5e1;
  }
  .ai-box h3 { color: #a78bfa; font-size: 0.9rem; margin: 0.9rem 0 0.3rem; font-family: 'Syne', sans-serif; }
  .ai-box strong { color: #e2e8f0; }
  .ai-box code { background: #1e2d45; padding: 0.1rem 0.35rem; border-radius: 3px;
                  font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #00d4ff; }
  .ai-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #7c3aed; margin-bottom: 0.5rem;
  }

  /* Streamlit widget overrides */
  .stSelectbox > div > div,
  .stNumberInput > div > div > input,
  .stTextArea textarea {
    background: #0a0e1a !important;
    border-color: #1e2d45 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
  }

  .stButton > button {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600; font-size: 0.78rem;
    border-radius: 7px; border: none;
    padding: 0.45rem 1rem;
  }
  .stButton > button[kind="primary"] {
    background: #00d4ff; color: #000;
  }
  .stButton > button[kind="secondary"] {
    background: transparent; border: 1px solid #1e2d45; color: #e2e8f0;
  }

  div[data-testid="metric-container"] {
    background: #0a0e1a; border: 1px solid #1e2d45;
    border-radius: 8px; padding: 0.5rem 0.75rem;
  }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.65rem !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'JetBrains Mono', monospace !important; }

  .stTabs [data-baseweb="tab-list"] { background: #111827; border-radius: 8px; gap: 0; }
  .stTabs [data-baseweb="tab"] { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #64748b; }
  .stTabs [aria-selected="true"] { color: #00d4ff !important; }

  hr { border-color: #1e2d45; }

  .week-card {
    background: #111827; border: 1px solid #1e2d45;
    border-radius: 10px; padding: 1rem 1.1rem;
    margin-bottom: 0.75rem; position: relative;
  }
  .week-card-improved { border-left: 3px solid #10b981 !important; }
  .week-card-same     { border-left: 3px solid #475569 !important; }
  .week-card-best     { border-left: 3px solid #f59e0b !important; }

  .week-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.6rem;
  }
  .week-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; font-weight: 700;
    color: #e2e8f0; letter-spacing: 0.05em;
  }
  .week-delta-up   { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #10b981; font-weight: 600; }
  .week-delta-down { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #ef4444; font-weight: 600; }
  .week-delta-new  { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b; }

  .week-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
    margin-bottom: 0.6rem;
  }
  .week-cell {
    background: #0a0e1a; border: 1px solid #1e2d4560;
    border-radius: 6px; padding: 0.35rem 0.6rem;
  }
  .week-cell-label { font-size: 0.58rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; font-family: 'JetBrains Mono', monospace; }
  .week-cell-value { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #cbd5e1; margin-top: 0.1rem; word-break: break-all; }
  .week-cell-value-hi { color: #f59e0b !important; font-weight: 600; }

  .week-reflection {
    background: #0a0e1a; border: 1px solid #1e2d45;
    border-radius: 6px; padding: 0.65rem 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; line-height: 1.65; color: #94a3b8;
    margin-top: 0.5rem;
  }
  .week-ai-badge {
    display: inline-block; padding: 0.12rem 0.45rem;
    border-radius: 4px; font-size: 0.62rem; font-weight: 600;
    background: rgba(124,58,237,0.15); color: #a78bfa;
    font-family: 'JetBrains Mono', monospace; margin-left: 0.4rem;
  }

  .header-title {
    font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(135deg, #00d4ff, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
  }
  .header-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #64748b; margin-top: 0.2rem;
  }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# FUNCTION CONFIGURATION (identical to capstone_optimiser.py)
# =============================================================================
FN_COLORS = [
    "#00d4ff","#7c3aed","#10b981","#f59e0b",
    "#ef4444","#ec4899","#8b5cf6","#06b6d4"
]

FUNCTION_CONFIG = {
    1: {
        "dims": 2, "bounds": [(0.0,1.0)]*2,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.0, "xi": 0.05,
        "y_transform": "arcsinh",
        "description": "2D contamination/radiation field",
        "dim_labels": ["x₁ (position)", "x₂ (position)"],
        "notes": "Sparse non-zero regions suggest the signal is localised. UCB with beta=2.0 keeps exploration high early to avoid missing the signal entirely. Reduce beta to 1.0 after a non-zero reading. arcsinh Y-transform enabled: spreads near-zero values so the GP can learn from tiny magnitude differences.",
    },
    2: {
        "dims": 2, "bounds": [(0.0,1.0)]*2,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.5, "xi": 0.1,
        "y_transform": "standardize",
        "heteroscedastic": True,
        "description": "2D noisy black-box log-likelihood",
        "dim_labels": ["x₁", "x₂"],
        "notes": "Explicitly noisy with local optima. Matern kernel handles rougher functions. High beta UCB resists premature exploitation of noisy early readings. Heteroscedastic GP enabled: per-point noise estimated via LOO residuals + kernel smoothing. The peak region near [0.70, 0.93] shows σ≈0.06 variation between neighbouring points — the het-GP assigns higher noise there, preventing the acquisition function from chasing noise-driven apparent gradients.",
    },
    3: {
        "dims": 3, "bounds": [(0.0,1.0)]*3,
        "kernel": "matern", "acquisition": "ei", "beta": 1.96, "xi": 0.02,
        "y_transform": "standardize",
        "description": "3D drug compound combinations",
        "dim_labels": ["Compound A", "Compound B", "Compound C"],
        "notes": "Physical process — Matern kernel appropriate. EI balances improvement focus with uncertainty. Negated side effects = maximise.",
    },
    4: {
        "dims": 4, "bounds": [(0.0,1.0)]*4,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.0, "xi": 0.05,
        "ard": True,
        "y_transform": "standardize",
        "description": "4D warehouse ML hyperparameter tuning",
        "dim_labels": ["Param 1", "Param 2", "Param 3", "Param 4"],
        "notes": "Rough landscape with local optima — stay exploratory early. Dynamic environment means old observations may drift in value. ARD enabled: P3 shows no significant correlation (r=−0.16, p=0.38) while P1/P4 dominate (r≈−0.50). ARD learns separate length-scales, effectively down-weighting P3.",
    },
    5: {
        "dims": 4, "bounds": [(0.0,1.0)]*4,
        "kernel": "rbf", "acquisition": "ei", "beta": 1.5, "xi": 0.01,
        "y_transform": "standardize",
        "description": "4D chemical yield (unimodal)",
        "dim_labels": ["Chemical 1", "Chemical 2", "Chemical 3", "Chemical 4"],
        "notes": "Unimodal — safest to exploit aggressively once in the right neighbourhood. Low xi on EI. Shift to pure exploitation by week 4.",
    },
    6: {
        "dims": 5, "bounds": [(0.0,1.0)]*5,
        "kernel": "matern", "acquisition": "ei", "beta": 1.96, "xi": 0.02,
        "y_transform": "standardize",
        "description": "5D cake recipe (negative penalty, max→0)",
        "dim_labels": ["Flour", "Sugar", "Eggs", "Butter", "Milk"],
        "notes": "Output is negative by design — scores near zero are best. Start near balanced ingredient ratios. Extreme values almost certainly score worse.",
    },
    7: {
        "dims": 6, "bounds": [(0.0,1.0)]*6,
        "kernel": "matern", "acquisition": "ei", "beta": 1.96, "xi": 0.05,
        "ard": True,
        "y_transform": "standardize",
        "description": "6D gradient boosting hyperparameters",
        "dim_labels": [
            "Dim 1: n_estimators [0–1]", "Dim 2: learning_rate [0–1]",
            "Dim 3: max_depth [0–1]",    "Dim 4: subsample [0–1]",
            "Dim 5: max_features [0–1]", "Dim 6: regularisation [0–1]",
        ],
        "notes": "Almost certainly GBM — all inputs normalised to [0,1] by the platform. EI appropriate given structured landscape. Key: learning_rate (dim2) and n_estimators (dim1) are inversely related. ARD enabled: known that dim1/dim2 dominate while dim5 (max_features) is less critical.",
        "informed_start": [0.333, 0.310, 0.250, 0.800, 0.800, 0.050],
    },
    8: {
        "dims": 8, "bounds": [(0.0,1.0)]*8,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.5, "xi": 0.1,
        "ard": True,
        "y_transform": "standardize",
        "description": "8D complex black-box (ML hyperparameters)",
        "dim_labels": [f"Param {i+1}" for i in range(8)],
        "notes": "Hardest function. GP at 8D will be uncertain — accept this. High beta UCB keeps exploration broad. ARD enabled: D1 and D3 have strong negative correlation (r≈−0.65) and dominate RF importance; ARD learns short length-scales for these vs long for D6/D8.",
    },
}

# =============================================================================
# PERSISTENCE
# =============================================================================
HISTORY_FILE = "capstone_history.json"
INITIAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "initial_data")


def load_initial_data() -> dict[int, dict]:
    """
    Load the provided initial .npy observations for each function.
    Returns {fn_id: {"X": ndarray (n, dims), "Y": ndarray (n,)}}
    Only loads files that actually exist; missing directories are silently skipped.
    """
    result = {}
    for fn_id in range(1, 9):
        folder = os.path.join(INITIAL_DATA_DIR, f"function_{fn_id}")
        x_path = os.path.join(folder, "initial_inputs.npy")
        y_path = os.path.join(folder, "initial_outputs.npy")
        if os.path.isfile(x_path) and os.path.isfile(y_path):
            X = np.load(x_path).astype(np.float64)
            Y = np.load(y_path).astype(np.float64).ravel()
            n = min(len(X), len(Y))
            result[fn_id] = {"X": X[:n], "Y": Y[:n]}
    return result


def align_xy_pair(fn_h: dict) -> bool:
    """
    Ensure X, Y and meta all have the same length.
    Truncates the longer lists. Returns True if anything was changed.
    """
    X, Y = fn_h.get("X", []), fn_h.get("Y", [])
    nx, ny = len(X), len(Y)
    changed = nx != ny
    n = min(nx, ny)
    fn_h["X"] = X[:n]
    fn_h["Y"] = Y[:n]
    w = fn_h.get("week", 0)
    if w > n:
        fn_h["week"] = n
    meta = fn_h.get("meta", [])
    if len(meta) != n:
        fn_h["meta"] = (meta + [{}] * n)[:n]
        changed = True
    return changed


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            raw = json.load(f)
        history = {int(k): v for k, v in raw.items()}
        dirty = False
        for _fid, fn_h in history.items():
            if "meta" not in fn_h:
                fn_h["meta"] = [{}] * len(fn_h.get("Y", []))
                dirty = True
            if "ai_analyses" not in fn_h:
                fn_h["ai_analyses"] = []
                dirty = True
            if align_xy_pair(fn_h):
                dirty = True
        if dirty:
            try:
                with open(HISTORY_FILE, "w") as f:
                    json.dump(history, f, indent=2)
            except OSError:
                pass
        return history
    return {i: {"X": [], "Y": [], "week": 0, "meta": [], "ai_analyses": []} for i in range(1, 9)}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# =============================================================================
# GP + ACQUISITION
# =============================================================================

# --- Y-transform helpers (applied in suggest_next; visualization uses raw Y) ---
def _yt_scale(Y: np.ndarray) -> float:
    """Robust IQR-based scale for arcsinh transform, bounded away from 0."""
    q75, q25 = np.percentile(Y, [75, 25])
    s = (q75 - q25) / 1.35  # ≈ std for Normal
    return float(max(s, np.abs(Y).mean() * 0.1, 1e-10))

def apply_y_transform(Y: np.ndarray, method: str | None, scale=None):
    """Transform Y before GP fitting. Returns (Y_t, scale_used).

    arcsinh:    sinh⁻¹(Y/s) — symmetric log-like, defined for all reals.
                Spreads near-zero values that differ only in tiny magnitude,
                which is the failure mode for F1 where all outputs ≈ 0.
    log1p:      log(1+Y) — for all-positive Y (monotone, reduces right skew).
    standardize: z-score: (Y − μ) / σ — zero mean, unit variance.
                scale is stored as (μ, σ) tuple so the transform is invertible.
                Ensures EI/UCB are computed in consistent units across functions
                with very different Y ranges (e.g. F4 [−33, −1] vs F5 [50, 1374]).
    """
    if method == "arcsinh":
        s = scale if scale is not None else _yt_scale(Y)
        return np.arcsinh(Y / s), s
    if method == "log1p":
        s = 1.0
        return np.log1p(np.clip(Y, 0, None)), s
    if method == "standardize":
        if scale is not None:
            y_mean, y_std = scale
        else:
            y_mean = float(Y.mean())
            y_std  = float(max(float(Y.std()), 1e-8))
        return (Y - y_mean) / y_std, (y_mean, y_std)
    return Y.copy(), 1.0

def invert_y_transform(Y_t: np.ndarray, method: str | None, scale) -> np.ndarray:
    """Inverse of apply_y_transform — restores original Y scale for display."""
    if method == "arcsinh":
        return np.sinh(Y_t) * scale
    if method == "log1p":
        return np.expm1(Y_t)
    if method == "standardize":
        y_mean, y_std = scale
        return Y_t * y_std + y_mean
    return Y_t


def build_gp(kernel_type: str = "matern", dims: int = 1, ard: bool = False,
             normalize_y: bool = True, alpha=1e-6):
    """Build a GP with isotropic or ARD kernel.

    ard=True: use a separate length-scale per dimension (Automatic Relevance
    Determination). The GP marginal-likelihood optimizer then assigns short
    length-scales to sensitive dimensions and long ones to irrelevant dims,
    effectively learning which inputs matter.

    normalize_y: set to False when Y has already been transformed by
    apply_y_transform (e.g. standardize, arcsinh) to avoid double-normalising.
    When no y_transform is active, True is the safe default.

    alpha: float or array-like (n_samples,). When an array is passed (from
    compute_heteroscedastic_alpha) each training point gets its own noise
    variance, giving a heteroscedastic GP.
    """
    ls = np.ones(dims) if (ard and dims > 1) else 1.0
    if kernel_type == "rbf":
        kernel = C(1.0) * RBF(length_scale=ls, length_scale_bounds=(1e-2, 10.0))
    else:
        kernel = C(1.0) * Matern(length_scale=ls, nu=2.5, length_scale_bounds=(1e-2, 10.0))
    return GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=normalize_y,
                                    n_restarts_optimizer=5)


def compute_heteroscedastic_alpha(
    X: np.ndarray,
    Y: np.ndarray,
    base_alpha: float = 1e-4,
    bandwidth: float = 0.20,
) -> np.ndarray:
    """Per-point noise variance via leave-one-out (LOO) residuals + kernel smooth.

    Algorithm
    ---------
    1. For each training point i, fit a GP on the remaining n-1 points and
       predict Y_i. The squared prediction error is an out-of-sample noise
       estimate for that region of the input space.
    2. Apply a Gaussian kernel smoother (bandwidth in [0,1] input units) to
       spread the estimate to neighbouring points, avoiding single-spike alphas.
    3. Clip to base_alpha so the GP kernel matrix remains numerically PSD.

    The returned array is ready to pass as the `alpha` argument of
    GaussianProcessRegressor.  It should be computed on the *transformed*
    Y_fit (e.g. after standardise) so its units match the GP target space.

    Falls back to a constant array when n < 4 (LOO unreliable with so few pts).
    """
    n = len(X)
    if n < 4:
        return np.full(n, base_alpha)

    loo_res_sq = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        kernel_loo = C(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp_loo = GaussianProcessRegressor(
            kernel=kernel_loo,
            alpha=base_alpha,
            normalize_y=True,
            n_restarts_optimizer=1,
        )
        gp_loo.fit(X[mask], Y[mask])
        pred_i = float(gp_loo.predict(X[i : i + 1])[0])
        loo_res_sq[i] = (float(Y[i]) - pred_i) ** 2

    # Gaussian kernel smoother: each point's alpha is a weighted average of
    # its neighbours' LOO residuals, so nearby noisy points raise each other's
    # noise estimate rather than creating isolated spikes.
    alphas = np.zeros(n)
    h2 = 2.0 * bandwidth ** 2
    for i in range(n):
        d2 = np.sum((X - X[i]) ** 2, axis=1)
        w = np.exp(-d2 / h2)
        w /= w.sum() + 1e-12
        alphas[i] = float(np.dot(w, loo_res_sq))

    return np.clip(alphas, base_alpha, None)


def compute_acquisition(acq, mean, std, y_max, beta, xi):
    if acq == "ucb":
        return mean + beta * std
    if acq == "variance":
        return std ** 2
    if acq == "ei":
        imp = mean - y_max - xi
        z = imp / (std + 1e-12)
        return imp * norm.cdf(z) + std * norm.pdf(z)
    if acq == "pi":
        z = (mean - y_max - xi) / (std + 1e-12)
        return norm.cdf(z)
    if acq == "mean":
        # Pure exploitation: argmax of GP posterior mean (no exploration bonus).
        # Appropriate for unimodal functions (e.g. F5) once the peak region is known.
        return mean
    return mean

def suggest_next(fn_id, history, acq_override=None, beta_override=None, xi_override=None,
                 initial_data: dict | None = None):
    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]
    acq  = acq_override  or cfg["acquisition"]
    beta = beta_override or cfg["beta"]
    xi   = xi_override   or cfg["xi"]

    # --- Assemble training data: initial .npy observations + portal submissions ---
    X_parts, Y_parts = [], []

    init = (initial_data or {}).get(fn_id)
    if init is not None:
        X_parts.append(init["X"])
        Y_parts.append(init["Y"])

    if fn_h["X"]:
        if align_xy_pair(fn_h):
            try:
                save_history(history)
            except OSError:
                pass
        X_sub = np.asarray(fn_h["X"], dtype=np.float64)
        Y_sub = np.asarray(fn_h["Y"], dtype=np.float64)
        n = min(len(X_sub), len(Y_sub))
        X_parts.append(X_sub[:n])
        Y_parts.append(Y_sub[:n])

    if not X_parts:
        if "informed_start" in cfg:
            return cfg["informed_start"], None, None
        return [round(np.random.uniform(0, 1), 6) for _ in range(cfg["dims"])], None, None

    X = np.vstack(X_parts)
    Y = np.concatenate(Y_parts)

    if not np.all(np.isfinite(Y)):
        Y = np.nan_to_num(Y, nan=0.0, posinf=1e300, neginf=-1e300)
    # Extreme magnitudes (e.g. typo 1e-185 or huge scores) destabilise the GP kernel matrix.
    Y = np.clip(Y, -1e12, 1e12)

    # Optional Y-transform (e.g. arcsinh for F1's near-zero landscape)
    y_transform = cfg.get("y_transform")
    Y_fit, yt_scale = apply_y_transform(Y, y_transform)
    y_max_fit = Y_fit.max()

    # Heteroscedastic GP: compute per-point noise from LOO residuals on Y_fit
    # (already standardised), then build the GP with that alpha array.
    # normalize_y must be False here — Y_fit is already in z-score units and
    # the alpha values are in the same z-score units; double-normalising would
    # divide both the targets and alphas by an extra factor.
    if cfg.get("heteroscedastic") and len(Y_fit) >= 4:
        alpha_arr = compute_heteroscedastic_alpha(X, Y_fit)
        gp = build_gp(cfg["kernel"], dims=cfg["dims"], ard=cfg.get("ard", False),
                      normalize_y=False, alpha=alpha_arr)
    else:
        # Don't double-normalise: if a y_transform is active it already
        # standardises Y; only fall back to GP-internal normalisation when
        # no transform is in use.
        gp = build_gp(cfg["kernel"], dims=cfg["dims"], ard=cfg.get("ard", False),
                      normalize_y=(y_transform is None))
    gp.fit(X, Y_fit)

    n_cand = 5000
    candidates = np.random.uniform(0, 1, (n_cand, cfg["dims"]))
    mean, std = gp.predict(candidates, return_std=True)
    scores = compute_acquisition(acq, mean, std, y_max_fit, beta, xi)
    best = np.argmax(scores)
    suggestion = np.clip(candidates[best], 0.0, 1.0)

    # Return mean/std in original Y scale for display.
    # yt_scale is a (mean, std) tuple for "standardize", a float for "arcsinh".
    mean_display = float(invert_y_transform(np.array([mean[best]]), y_transform, yt_scale)[0])
    if y_transform == "arcsinh":
        std_display = float(std[best] * yt_scale)
    elif y_transform == "standardize":
        std_display = float(std[best] * yt_scale[1])   # scale back by Y std-dev
    else:
        std_display = float(std[best])
    return [round(float(v), 6) for v in suggestion], mean_display, std_display

# =============================================================================
# CHART HELPERS
# =============================================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(10,14,26,1)",
    font=dict(family="JetBrains Mono, monospace", color="#64748b", size=11),
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
    yaxis=dict(gridcolor="#1e2d45", zerolinecolor="#1e2d45"),
)

def make_history_chart(fn_id, color, fn_h, initial_data=None):
    Y = fn_h["Y"]
    if not Y:
        return None
    weeks = list(range(1, len(Y)+1))
    y_max = max(Y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weeks, y=Y, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(color=color, size=7, line=dict(color="#0a0e1a", width=1.5)),
        fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
        name="Portal submissions",
    ))
    # Initial data best as a reference baseline
    init = (initial_data or {}).get(fn_id)
    if init is not None:
        init_best = float(init["Y"].max())
        fig.add_hline(
            y=init_best, line_dash="dot", line_color="#475569", line_width=1.5,
            annotation_text=f"Initial best: {init_best:.4f}",
            annotation_font_color="#94a3b8", annotation_font_size=9,
            annotation_position="bottom right",
        )
    fig.add_hline(y=y_max, line_dash="dash", line_color="#f59e0b", line_width=1,
                  annotation_text=f"Portal best: {y_max:.4f}", annotation_font_color="#f59e0b",
                  annotation_font_size=10)
    fig.update_layout(**PLOTLY_LAYOUT, height=220,
                      xaxis_title="Week", yaxis_title="Output",
                      showlegend=False)
    fig.update_xaxes(tickvals=weeks, dtick=1)
    return fig

def _prepare_gp(fn_id, history, initial_data):
    """
    Fit a GP on combined initial + portal data for fn_id.
    Returns (gp, X_train, Y_train, best_x) or None if < 2 points.
    """
    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]
    X_parts, Y_parts = [], []
    init = (initial_data or {}).get(fn_id)
    if init is not None:
        X_parts.append(init["X"])
        Y_parts.append(init["Y"])
    if fn_h["X"]:
        align_xy_pair(fn_h)
        Xs = np.asarray(fn_h["X"], dtype=np.float64)
        Ys = np.asarray(fn_h["Y"], dtype=np.float64)
        n = min(len(Xs), len(Ys))
        X_parts.append(Xs[:n])
        Y_parts.append(Ys[:n])
    if not X_parts:
        return None
    X = np.vstack(X_parts)
    Y = np.concatenate(Y_parts)
    if not np.all(np.isfinite(Y)):
        Y = np.nan_to_num(Y, nan=0.0)
    Y = np.clip(Y, -1e12, 1e12)
    if len(Y) < 2:
        return None
    # Visualization uses raw Y (no transform) so plots stay in interpretable units.
    # For heteroscedastic functions, convert LOO alpha from raw-Y² units to
    # the normalized units that the GP uses internally (normalize_y=True divides
    # targets by Y.std(), so the kernel operates on a unit-variance signal and
    # alpha must be scaled by 1/Y.std()² to match).
    if cfg.get("heteroscedastic") and len(Y) >= 4:
        y_std = max(float(np.std(Y)), 1e-8)
        alpha_raw = compute_heteroscedastic_alpha(X, Y)
        alpha_norm = alpha_raw / (y_std ** 2)
        gp = build_gp(cfg["kernel"], dims=cfg["dims"], ard=cfg.get("ard", False),
                      normalize_y=True, alpha=alpha_norm)
    else:
        gp = build_gp(cfg["kernel"], dims=cfg["dims"], ard=cfg.get("ard", False))
    gp.fit(X, Y)
    best_x = X[np.argmax(Y)].copy()
    return gp, X, Y, best_x


def make_gp_slice_plot(fn_id, color, history, initial_data, suggestion=None):
    """
    For each input dimension, plot a 1D slice through the GP posterior
    (all other dims fixed at the best-known point), showing mean ± 95% CI.
    Initial data and portal submissions are shown with distinct colours.
    An optional vertical line marks the current suggestion per dimension.
    """
    result = _prepare_gp(fn_id, history, initial_data)
    if result is None:
        return None
    gp, X_train, Y_train, best_x = result
    cfg = FUNCTION_CONFIG[fn_id]
    dims = cfg["dims"]
    labels = [lbl.split(":")[0].strip() for lbl in cfg["dim_labels"]]

    # Split training data into initial vs portal submissions
    init = (initial_data or {}).get(fn_id)
    n_init = len(init["X"]) if init is not None else 0

    n_cols = 2
    n_rows = math.ceil(dims / n_cols)
    subplot_titles = [labels[d] if d < dims else "" for d in range(n_rows * n_cols)]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    grid = np.linspace(0, 1, 150)
    rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

    for d in range(dims):
        row, col = divmod(d, n_cols)
        row += 1; col += 1

        X_slice = np.tile(best_x, (len(grid), 1))
        X_slice[:, d] = grid
        mean, std = gp.predict(X_slice, return_std=True)
        ci_upper = mean + 1.96 * std
        ci_lower = mean - 1.96 * std

        # 95% CI band
        fig.add_trace(go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([ci_upper, ci_lower[::-1]]),
            fill="toself",
            fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=(d == 0),
            legendgroup="ci",
        ), row=row, col=col)

        # GP posterior mean
        fig.add_trace(go.Scatter(
            x=grid, y=mean,
            mode="lines", line=dict(color=color, width=2),
            name="GP Mean", showlegend=(d == 0),
            legendgroup="mean",
        ), row=row, col=col)

        # Initial data observations (muted grey)
        if n_init > 0:
            fig.add_trace(go.Scatter(
                x=X_train[:n_init, d], y=Y_train[:n_init],
                mode="markers",
                marker=dict(color="#475569", size=5, opacity=0.65,
                            symbol="circle", line=dict(color="#0a0e1a", width=0.5)),
                name="Initial data", showlegend=(d == 0),
                legendgroup="init",
            ), row=row, col=col)

        # Portal submissions (function colour + amber for best)
        if len(X_train) > n_init:
            X_portal = X_train[n_init:]
            Y_portal = Y_train[n_init:]
            best_p = int(np.argmax(Y_portal))
            non_best = [i for i in range(len(Y_portal)) if i != best_p]
            if non_best:
                fig.add_trace(go.Scatter(
                    x=X_portal[non_best, d], y=Y_portal[non_best],
                    mode="markers",
                    marker=dict(color=color, size=7, opacity=0.9,
                                symbol="circle", line=dict(color="#0a0e1a", width=1)),
                    name="Your submissions", showlegend=(d == 0),
                    legendgroup="portal",
                ), row=row, col=col)
            fig.add_trace(go.Scatter(
                x=[X_portal[best_p, d]], y=[Y_portal[best_p]],
                mode="markers",
                marker=dict(color="#f59e0b", size=10, opacity=1.0,
                            symbol="diamond", line=dict(color="#0a0e1a", width=1.5)),
                name="Best submission", showlegend=(d == 0),
                legendgroup="portal_best",
            ), row=row, col=col)

        # Current suggestion as a vertical dashed line
        if suggestion and d < len(suggestion):
            fig.add_vline(
                x=suggestion[d], line_color="#00d4ff", line_dash="dash",
                line_width=1.5, row=row, col=col,
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=220 * n_rows,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=10, color="#94a3b8"),
        ),
    )
    fig.update_annotations(font=dict(size=10, color="#94a3b8"))
    fig.update_xaxes(range=[0, 1], tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


def make_acq_comparison_plot(fn_id, history, initial_data, beta, xi):
    """
    For each input dimension, show all four acquisition functions (UCB, EI, PI, Variance)
    normalised to [0,1] so their shapes can be compared side by side.
    """
    result = _prepare_gp(fn_id, history, initial_data)
    if result is None:
        return None
    gp, X_train, Y_train, best_x = result
    cfg = FUNCTION_CONFIG[fn_id]
    dims = cfg["dims"]
    labels = [lbl.split(":")[0].strip() for lbl in cfg["dim_labels"]]
    y_max = float(Y_train.max())

    n_cols = 2
    n_rows = math.ceil(dims / n_cols)
    subplot_titles = [labels[d] if d < dims else "" for d in range(n_rows * n_cols)]
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12, horizontal_spacing=0.10,
    )

    grid = np.linspace(0, 1, 150)
    acq_styles = {
        "UCB":      dict(color="#00d4ff", dash="solid"),
        "EI":       dict(color="#10b981", dash="solid"),
        "PI":       dict(color="#a78bfa", dash="solid"),
        "Variance": dict(color="#f59e0b", dash="dot"),
    }

    def _norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-12)

    for d in range(dims):
        row, col = divmod(d, n_cols)
        row += 1; col += 1

        X_slice = np.tile(best_x, (len(grid), 1))
        X_slice[:, d] = grid
        mean, std = gp.predict(X_slice, return_std=True)

        scores = {
            "UCB":      _norm(mean + beta * std),
            "EI":       _norm(compute_acquisition("ei", mean, std, y_max, beta, xi)),
            "PI":       _norm(compute_acquisition("pi", mean, std, y_max, beta, xi)),
            "Variance": _norm(std ** 2),
        }

        for acq_name, vals in scores.items():
            sty = acq_styles[acq_name]
            fig.add_trace(go.Scatter(
                x=grid, y=vals,
                mode="lines",
                line=dict(color=sty["color"], width=1.8, dash=sty["dash"]),
                name=acq_name,
                showlegend=(d == 0),
                legendgroup=acq_name,
            ), row=row, col=col)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=220 * n_rows,
        yaxis_title="Score (normalised)",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            font=dict(size=10, color="#94a3b8"),
        ),
    )
    fig.update_annotations(font=dict(size=10, color="#94a3b8"))
    fig.update_xaxes(range=[0, 1], tickfont=dict(size=9))
    fig.update_yaxes(range=[-0.05, 1.05], tickfont=dict(size=9))
    return fig


def make_query_space_plot(fn_id, color, history, initial_data, suggestion=None):
    """
    Strip plot showing, for each input dimension, where the current suggestion
    sits relative to all historical observations.

    Layers (bottom → top):
      - Initial data: grey circles
      - Initial best: grey diamond
      - Portal submissions: function-colour circles
      - Best portal submission: amber diamond
      - Next query suggestion: cyan star
    """
    cfg = FUNCTION_CONFIG[fn_id]
    dims = cfg["dims"]
    labels = [lbl.split(":")[0].strip() for lbl in cfg["dim_labels"]]
    init = (initial_data or {}).get(fn_id)
    fn_h = history[fn_id]

    fig = go.Figure()

    # --- Initial data ---
    if init is not None:
        X_init = init["X"]
        Y_init = init["Y"]
        best_init_idx = int(np.argmax(Y_init))

        xs, ys, texts = [], [], []
        for d in range(dims):
            for j in range(len(X_init)):
                xs.append(X_init[j, d])
                ys.append(labels[d])
                texts.append(f"Initial #{j+1}: {labels[d]} = {X_init[j,d]:.4f}  (Y={Y_init[j]:.4g})")
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(color="#334155", size=6, opacity=0.7,
                        symbol="circle", line=dict(color="#0a0e1a", width=0.5)),
            name="Initial data",
            text=texts, hoverinfo="text",
            legendgroup="init",
        ))

        # Best initial point
        xs_b, ys_b, texts_b = [], [], []
        for d in range(dims):
            xs_b.append(X_init[best_init_idx, d])
            ys_b.append(labels[d])
            texts_b.append(f"Initial best: {labels[d]} = {X_init[best_init_idx,d]:.4f}  (Y={Y_init[best_init_idx]:.4g})")
        fig.add_trace(go.Scatter(
            x=xs_b, y=ys_b, mode="markers",
            marker=dict(color="#94a3b8", size=9, opacity=0.9,
                        symbol="diamond", line=dict(color="#0a0e1a", width=1)),
            name="Initial best",
            text=texts_b, hoverinfo="text",
            legendgroup="init_best",
        ))

    # --- Portal observations ---
    if fn_h["X"]:
        X_portal = np.asarray(fn_h["X"])
        Y_portal = np.asarray(fn_h["Y"])
        best_p = int(np.argmax(Y_portal))
        non_best = [i for i in range(len(Y_portal)) if i != best_p]

        if non_best:
            xs, ys, texts = [], [], []
            for i in non_best:
                for d in range(dims):
                    xs.append(X_portal[i, d])
                    ys.append(labels[d])
                    texts.append(f"Week {i+1}: {labels[d]} = {X_portal[i,d]:.4f}  (Y={Y_portal[i]:.4g})")
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(color=color, size=9, opacity=0.85,
                            symbol="circle", line=dict(color="#0a0e1a", width=1)),
                name="Your submissions",
                text=texts, hoverinfo="text",
                legendgroup="portal",
            ))

        # Best portal point
        xs_b, ys_b, texts_b = [], [], []
        for d in range(dims):
            xs_b.append(X_portal[best_p, d])
            ys_b.append(labels[d])
            texts_b.append(f"Week {best_p+1} (best): {labels[d]} = {X_portal[best_p,d]:.4f}  (Y={Y_portal[best_p]:.4g})")
        fig.add_trace(go.Scatter(
            x=xs_b, y=ys_b, mode="markers",
            marker=dict(color="#f59e0b", size=13, opacity=1.0,
                        symbol="diamond", line=dict(color="#0a0e1a", width=1.5)),
            name=f"Best submission (W{best_p+1})",
            text=texts_b, hoverinfo="text",
            legendgroup="portal_best",
        ))

    # --- Current suggestion ---
    if suggestion:
        xs_s, ys_s, texts_s = [], [], []
        for d in range(dims):
            xs_s.append(suggestion[d])
            ys_s.append(labels[d])
            texts_s.append(f"Next query: {labels[d]} = {suggestion[d]:.6f}")
        fig.add_trace(go.Scatter(
            x=xs_s, y=ys_s, mode="markers",
            marker=dict(color="#00d4ff", size=14, symbol="star",
                        line=dict(color="#0a0e1a", width=1.5)),
            name="Next query (suggested)",
            text=texts_s, hoverinfo="text",
            legendgroup="suggestion",
        ))

    layout = {
        **PLOTLY_LAYOUT,
        "height": max(200, 38 * dims + 110),
        "legend": dict(
            orientation="h", yanchor="bottom", y=1.04,
            font=dict(size=10, color="#94a3b8"),
            itemclick=False, itemdoubleclick=False,
        ),
        "margin": dict(l=110, r=20, t=55, b=40),
        "hovermode": "closest",
    }
    fig.update_layout(**layout)
    fig.update_xaxes(range=[-0.05, 1.05], title_text="Input value [0, 1]",
                     gridcolor="#1e2d45", zerolinecolor="#1e2d45")
    fig.update_yaxes(autorange="reversed", gridcolor="#1e2d45")
    return fig


def ai_analysis(fn_id, history, initial_data, api_key: str) -> str:
    """
    Call Claude to analyse observations for fn_id and return strategy recommendations.
    """
    if not HAS_ANTHROPIC:
        return "Install the `anthropic` package (`pip install anthropic`) to enable AI analysis."

    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]
    init = (initial_data or {}).get(fn_id)

    init_summary = "No initial data loaded."
    if init is not None:
        Y_init = init["Y"]
        init_summary = (
            f"{len(Y_init)} initial observations provided by the challenge.\n"
            f"  Y range: [{float(Y_init.min()):.4f}, {float(Y_init.max()):.4f}]  "
            f"  Mean: {float(Y_init.mean()):.4f}  Std: {float(Y_init.std()):.4f}\n"
            f"  Best initial X: {init['X'][int(np.argmax(Y_init))].tolist()}"
        )

    portal_obs = "No portal submissions recorded yet."
    if fn_h["Y"]:
        lines = []
        for i, (x, y) in enumerate(zip(fn_h["X"], fn_h["Y"])):
            m = fn_h.get("meta", [{}] * len(fn_h["Y"]))
            mi = m[i] if i < len(m) else {}
            acq_str = f"acq={mi.get('acq','?').upper()}, β={mi.get('beta','?')}, ξ={mi.get('xi','?')}, kernel={mi.get('kernel','?')}"
            lines.append(f"  Week {i+1}: X={[round(v,4) for v in x]}, Y={y:.6g}  [{acq_str}]")
        portal_obs = "\n".join(lines)

    best_y = max(fn_h["Y"]) if fn_h["Y"] else None
    best_str = f"{best_y:.6g}" if best_y is not None else "none yet"

    prompt = f"""You are an expert Bayesian optimisation advisor analysing progress on a black-box maximisation challenge.

## Function {fn_id} — {cfg['description']}
- Dimensionality: {cfg['dims']}D, all inputs normalised to [0, 1]
- Dimensions: {', '.join(cfg['dim_labels'])}
- Goal: MAXIMISE the output

## Initial Data (provided at challenge start)
{init_summary}

## Portal Submissions (one per week, each includes the acquisition settings used)
{portal_obs}

## Current Best Portal Output
{best_str}

## Domain Notes
{cfg['notes']}

---
Please provide:
1. **Landscape Analysis** — what does the data tell us about the shape of this function? Are there patterns, clusters, or ridgelines visible?
2. **Acquisition Strategy Review** — given the settings used so far (acq function, β, ξ, kernel), were they appropriate? What would you change and why?
3. **Next Query Recommendation** — suggest a specific region or point to query next and explain the reasoning.
4. **β and ξ Settings** — recommend specific values for the next submission with justification.
5. **Overall Strategy** — what is the best approach for the remaining queries to maximise the output?

Be concise and specific. Use markdown formatting."""

    try:
        client = _anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
    except Exception as exc:
        return f"API error: {exc}"


def make_summary_chart(history):
    rows = []
    for fn_id in range(1, 9):
        fn_h = history[fn_id]
        best = max(fn_h["Y"]) if fn_h["Y"] else None
        rows.append({
            "fn": f"F{fn_id}",
            "desc": FUNCTION_CONFIG[fn_id]["description"][:30],
            "best": best,
            "obs": len(fn_h["Y"]),
            "color": FN_COLORS[fn_id-1],
        })
    df = pd.DataFrame(rows)
    df_valid = df[df["best"].notna()]
    if df_valid.empty:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_valid["fn"], y=df_valid["best"],
        marker_color=df_valid["color"].tolist(),
        text=df_valid["best"].apply(lambda v: f"{v:.4f}"),
        textposition="outside", textfont=dict(size=10, color="#94a3b8"),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=240,
                      xaxis_title="", yaxis_title="Best Output",
                      showlegend=False)
    return fig

# =============================================================================
# REFLECTION GENERATOR
# =============================================================================
def generate_reflection(fn_id, history):
    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]
    if not fn_h["Y"]:
        return "Record at least one observation to generate your reflection."
    Y = fn_h["Y"]; X = fn_h["X"]
    y_max = max(Y); best_idx = Y.index(y_max)
    last_y = Y[-1]; last_x = X[-1]
    improved = (last_y == y_max and len(Y) > 1)
    acq = cfg["acquisition"]
    return f"""Week {fn_h['week']} Reflection — Function {fn_id}: {cfg['description']}

This week I submitted input [{', '.join(f'{v:.4f}' for v in last_x)}] and received an output of {last_y:.4f}.

{'This improved on the previous best, confirming the surrogate model prediction that this region was promising.' if improved else f"This did not improve on the current best of {y_max:.4f}. The observation has updated the GP posterior, reducing uncertainty in this region."}

Acquisition function: {acq.upper()} (β={cfg['beta']}, ξ={cfg['xi']})
{cfg['notes']}

Across {len(Y)} observation(s) to date, the best observed value is {y_max:.4f}
at input [{', '.join(f'{v:.4f}' for v in X[best_idx])}].

Next week I will {'continue exploiting this region, reducing exploration pressure.' if improved else 'explore a different region — this area does not appear to contain the global maximum.'}"""

# =============================================================================
# SESSION STATE INIT
# =============================================================================
if "history" not in st.session_state:
    st.session_state.history = load_history()
if "active_fn" not in st.session_state:
    st.session_state.active_fn = 1
if "suggestion" not in st.session_state:
    st.session_state.suggestion = {}
if "gp_info" not in st.session_state:
    st.session_state.gp_info = {}
if "acq_overrides" not in st.session_state:
    st.session_state.acq_overrides = {}
if "query_draft" not in st.session_state:
    st.session_state.query_draft = {}
if "initial_data" not in st.session_state:
    st.session_state.initial_data = load_initial_data()
if "ai_cache" not in st.session_state:
    # Pre-populate session cache from the most recent stored analysis for each function
    _cache: dict = {}
    for _fid in range(1, 9):
        _analyses = st.session_state.history[_fid].get("ai_analyses", [])
        if _analyses:
            _latest = _analyses[-1]
            _cache[_fid] = {
                "cache_key": (_fid, _latest.get("obs_count", 0)),
                "text": _latest["text"],
            }
    st.session_state.ai_cache = _cache

history = st.session_state.history
initial_data = st.session_state.initial_data

# Resolve Anthropic API key: secrets → env → None (prompts user in UI)
def _get_api_key() -> str | None:
    try:
        k = st.secrets["ANTHROPIC_API_KEY"]
        if k:
            return k
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("ANTHROPIC_API_KEY")

# Auto-generate week-1 suggestions for any function that has no observations yet
# and hasn't had a suggestion generated yet — so inputs are never blank on load
for _fn_id in range(1, 9):
    if _fn_id not in st.session_state.suggestion:
        _sug, _, _ = suggest_next(_fn_id, history, initial_data=initial_data)
        st.session_state.suggestion[_fn_id] = _sug
        st.session_state.gp_info[_fn_id] = (None, None)
    _dims = len(FUNCTION_CONFIG[_fn_id]["dim_labels"])
    _sug = st.session_state.suggestion[_fn_id]
    if _fn_id not in st.session_state.query_draft or len(st.session_state.query_draft[_fn_id]) != _dims:
        st.session_state.query_draft[_fn_id] = [
            float(_sug[i]) if i < len(_sug) else 0.5 for i in range(_dims)
        ]

# =============================================================================
# HEADER
# =============================================================================
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<div class="header-title">Capstone Optimiser</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-sub">Bayesian optimisation dashboard · 8 functions · 1 query / function / week</div>', unsafe_allow_html=True)
with col_h2:
    max_week = max((history[i].get("week", 0) for i in range(1, 9)), default=0)
    st.metric("Current Week", max_week if max_week > 0 else "—")

st.markdown("---")

# =============================================================================
# MAIN LAYOUT: left = function list, right = detail
# =============================================================================
left_col, right_col = st.columns([1, 2], gap="large")

# ---- LEFT: Function cards ----
with left_col:
    st.markdown('<div class="section-label">Functions</div>', unsafe_allow_html=True)
    for fn_id in range(1, 9):
        cfg = FUNCTION_CONFIG[fn_id]
        fn_h = history[fn_id]
        color = FN_COLORS[fn_id - 1]
        best = f"{max(fn_h['Y']):.4f}" if fn_h["Y"] else "—"
        is_active = st.session_state.active_fn == fn_id
        acq = st.session_state.acq_overrides.get(fn_id, cfg["acquisition"]).upper()
        _pill_map = {"ucb": "pill-ucb", "ei": "pill-ei", "pi": "pill-pi",
                     "variance": "pill-var", "mean": "pill-mean"}
        pill_cls = _pill_map.get(acq.lower(), "pill-var")

        card_cls = "fn-card fn-card-active" if is_active else "fn-card"
        st.markdown(f"""
        <div class="{card_cls}">
          <div class="fn-card-bar" style="background:{color}"></div>
          <div style="display:flex;align-items:center;margin-bottom:0.5rem">
            <span class="fn-number" style="background:{color}">{fn_id}</span>
            <div>
              <div class="fn-title">{cfg['description']}</div>
              <div class="fn-dims">{cfg['dims']}D → 1D</div>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.4rem;margin-bottom:0.5rem">
            <div style="background:#0a0e1a;border:1px solid #1e2d45;border-radius:6px;padding:0.3rem;text-align:center">
              <div class="stat-val" style="color:{color}">{fn_h.get('week',0)}</div>
              <div class="stat-lbl">Week</div>
            </div>
            <div style="background:#0a0e1a;border:1px solid #1e2d45;border-radius:6px;padding:0.3rem;text-align:center">
              <div class="stat-val" style="color:{color}">{len(fn_h['Y'])}</div>
              <div class="stat-lbl">Obs</div>
            </div>
            <div style="background:#0a0e1a;border:1px solid #1e2d45;border-radius:6px;padding:0.3rem;text-align:center">
              <div class="stat-val" style="color:{color}">{best}</div>
              <div class="stat-lbl">Best</div>
            </div>
          </div>
          <span class="pill {pill_cls}">{acq}</span>
          <span class="pill" style="background:rgba(124,58,237,0.15);color:#a78bfa">β={cfg['beta']}</span>
          <span class="pill" style="background:rgba(16,185,129,0.15);color:#10b981">ξ={cfg['xi']}</span>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"Select F{fn_id}", key=f"sel_{fn_id}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_fn = fn_id
            st.rerun()

# ---- RIGHT: Detail panel ----
with right_col:
    fn_id = st.session_state.active_fn
    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]
    color = FN_COLORS[fn_id - 1]

    st.markdown(
        f'<div class="section-label">Function {fn_id} — {cfg["description"]}</div>',
        unsafe_allow_html=True
    )

    tabs = st.tabs(["📥  Query", "📊  History", "📝  Reflection", "⚙️  Strategy", "🤖  AI Analysis", "📅  Journal"])

    # ---------------------------------------------------------------
    # TAB 1: QUERY
    # ---------------------------------------------------------------
    with tabs[0]:
        q_left, q_right = st.columns([1, 1], gap="medium")

        with q_left:
            st.markdown("**Input Values**")
            input_vals = []
            qd = st.session_state.query_draft
            dims = len(cfg["dim_labels"])
            if fn_id not in qd or len(qd[fn_id]) != dims:
                _sug = st.session_state.suggestion.get(fn_id, [])
                qd[fn_id] = [
                    float(_sug[i]) if i < len(_sug) else 0.5 for i in range(dims)
                ]

            applied_key = f"sug_applied_{fn_id}"
            current_sug = st.session_state.suggestion.get(fn_id, [])
            last_applied = st.session_state.get(applied_key, None)
            # New optimizer suggestion → sync draft + widget keys (before widgets exist).
            if current_sug is not None and len(current_sug) > 0 and str(current_sug) != last_applied:
                for j in range(dims):
                    qd[fn_id][j] = float(current_sug[j]) if j < len(current_sug) else qd[fn_id][j]
                    st.session_state[f"inp_{fn_id}_{j}"] = float(qd[fn_id][j])
                st.session_state[applied_key] = str(current_sug)

            for i, label in enumerate(cfg["dim_labels"]):
                wkey = f"inp_{fn_id}_{i}"
                # Widget keys are cleared when that function's inputs are not rendered
                # (switching active function). Restore from persistent draft.
                if wkey not in st.session_state:
                    st.session_state[wkey] = float(qd[fn_id][i])
                v = st.number_input(
                    label, min_value=0.0, max_value=1.0,
                    step=0.000001, format="%.6f",
                    key=wkey,
                )
                qd[fn_id][i] = float(v)
                input_vals.append(v)

            obs_output = st.number_input(
                "Observed Output (from portal)",
                value=0.0, step=0.0001, format="%.6f",
                key=f"obs_{fn_id}"
            )

        with q_right:
            st.markdown("**Acquisition Function**")
            ACQ_OPTIONS = ["ucb", "ei", "pi", "variance", "mean"]
            acq_choice = st.selectbox(
                "Method", ACQ_OPTIONS,
                index=ACQ_OPTIONS.index(
                    st.session_state.acq_overrides.get(fn_id, cfg["acquisition"])
                    if st.session_state.acq_overrides.get(fn_id, cfg["acquisition"]) in ACQ_OPTIONS
                    else cfg["acquisition"]
                ),
                key=f"acq_{fn_id}",
                label_visibility="collapsed",
                help="mean = pure exploitation (argmax GP posterior mean). Best for unimodal functions once the peak region is found.",
            )
            st.session_state.acq_overrides[fn_id] = acq_choice

            beta_val = st.slider("β (UCB exploration)", 0.5, 5.0, cfg["beta"], 0.1, key=f"beta_{fn_id}")
            xi_val   = st.slider("ξ (EI/PI exploration)", 0.0, 0.5, cfg["xi"], 0.01, key=f"xi_{fn_id}")

            st.markdown("")
            c1, c2 = st.columns(2)
            # New suggestion values must be applied in the left column *before*
            # number_input widgets are created (see sug_applied / inp_* logic there).
            # Setting st.session_state[inp_*] here would run after those widgets
            # exist in this script pass and triggers StreamlitAPIException.

            with c1:
                if st.button("🎯  Record + Next Query", key=f"rec_{fn_id}", type="primary", use_container_width=True):
                    fn_h["X"].append([round(v, 6) for v in input_vals])
                    fn_h["Y"].append(float(obs_output))
                    fn_h["week"] = fn_h.get("week", 0) + 1
                    if "meta" not in fn_h:
                        fn_h["meta"] = [{}] * (len(fn_h["Y"]) - 1)
                    fn_h["meta"].append({
                        "acq": acq_choice,
                        "beta": beta_val,
                        "xi": xi_val,
                        "kernel": cfg["kernel"],
                    })
                    st.session_state.ai_cache.pop(fn_id, None)
                    save_history(history)
                    suggestion, gp_mean, gp_std = suggest_next(
                        fn_id, history,
                        acq_override=acq_choice,
                        beta_override=beta_val,
                        xi_override=xi_val,
                        initial_data=initial_data,
                    )
                    st.session_state.suggestion[fn_id] = suggestion
                    st.session_state.gp_info[fn_id] = (gp_mean, gp_std)
                    st.session_state.pop(f"sug_applied_{fn_id}", None)
                    st.rerun()
            with c2:
                if st.button("🔍  Regenerate Week 1 Query", key=f"w1_{fn_id}", use_container_width=True):
                    suggestion, gp_mean, gp_std = suggest_next(
                        fn_id, history,
                        acq_override=acq_choice,
                        beta_override=beta_val,
                        xi_override=xi_val,
                        initial_data=initial_data,
                    )
                    st.session_state.suggestion[fn_id] = suggestion
                    st.session_state.gp_info[fn_id] = (gp_mean, gp_std)
                    st.session_state.pop(f"sug_applied_{fn_id}", None)
                    st.rerun()

            # Suggestion output
            if fn_id in st.session_state.suggestion:
                sug = st.session_state.suggestion[fn_id]
                gp_info = st.session_state.gp_info.get(fn_id)
                st.markdown(f"""
                <div class="suggestion-box">
                  <div class="suggestion-label">Submit to portal — Function {fn_id}</div>
                  <div class="suggestion-value">[{', '.join(f'{v:.6f}' for v in sug)}]</div>
                </div>
                """, unsafe_allow_html=True)
                if gp_info and gp_info[0] is not None:
                    m, s = gp_info
                    yt = cfg.get("y_transform")
                    ard_on = cfg.get("ard", False)
                    het_on = cfg.get("heteroscedastic", False)
                    badges = ""
                    if yt:
                        yt_color = "#f87171" if yt == "arcsinh" else "#fbbf24"
                        yt_bg    = "rgba(239,68,68,0.12)" if yt == "arcsinh" else "rgba(251,191,36,0.12)"
                        badges += f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;background:{yt_bg};color:{yt_color};border-radius:3px;padding:0.1rem 0.35rem;margin-left:0.4rem">{yt}</span>'
                    if ard_on:
                        badges += '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;background:rgba(16,185,129,0.12);color:#10b981;border-radius:3px;padding:0.1rem 0.35rem;margin-left:0.3rem">ARD</span>'
                    if het_on:
                        badges += '<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.6rem;background:rgba(139,92,246,0.12);color:#8b5cf6;border-radius:3px;padding:0.1rem 0.35rem;margin-left:0.3rem">het-GP</span>'
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem">
                      <div style="background:#0a0e1a;border:1px solid #1e2d45;border-radius:6px;padding:0.4rem;text-align:center">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#00d4ff;font-weight:600">{m:.4f}</div>
                        <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em">GP Mean{badges}</div>
                      </div>
                      <div style="background:#0a0e1a;border:1px solid #1e2d45;border-radius:6px;padding:0.4rem;text-align:center">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#7c3aed;font-weight:600">{s:.4f}</div>
                        <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em">GP Std</div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Observation log
            if fn_h["Y"]:
                st.markdown("")
                st.markdown('<div class="section-label" style="margin-top:0.5rem">Observation Log</div>', unsafe_allow_html=True)
                y_max_val = max(fn_h["Y"])
                fn_meta = fn_h.get("meta", [])
                rows_html = ""
                for i, (x, y) in enumerate(zip(fn_h["X"], fn_h["Y"])):
                    is_best = y == y_max_val
                    out_cls = "obs-best" if is_best else "obs-output"
                    star = " ★" if is_best else ""
                    mi = fn_meta[i] if i < len(fn_meta) else {}
                    meta_str = ""
                    if mi:
                        meta_str = f"{mi.get('acq','?').upper()} β={mi.get('beta','?')} ξ={mi.get('xi','?')}"
                    rows_html += f"""
                    <div class="obs-row">
                      <div class="obs-week">W{i+1}</div>
                      <div class="obs-input">[{', '.join(f'{v:.3f}' for v in x)}]</div>
                      <div class="{out_cls}">{y:.4f}{star}</div>
                      <div class="obs-meta">{meta_str}</div>
                    </div>"""
                st.markdown(f'<div style="max-height:180px;overflow-y:auto">{rows_html}</div>', unsafe_allow_html=True)

                # Delete last observation
                if st.button("↩ Undo last observation", key=f"undo_{fn_id}"):
                    fn_h["X"].pop(); fn_h["Y"].pop()
                    fn_h["week"] = max(0, fn_h.get("week", 1) - 1)
                    if fn_h.get("meta"):
                        fn_h["meta"].pop()
                    st.session_state.ai_cache.pop(fn_id, None)
                    save_history(history)
                    st.rerun()

        # Full-width input space plot — always visible once initial data is loaded
        st.markdown("")
        st.markdown('<div class="section-label">Input Space — This Week\'s Query vs All Observations</div>', unsafe_allow_html=True)
        st.caption("★ Cyan star = suggested next query · ◆ Amber diamond = best submission · ◆ Grey diamond = initial data best · Grey dots = initial data · Coloured dots = your submissions")
        fig_space = make_query_space_plot(
            fn_id, color, history, initial_data,
            suggestion=st.session_state.suggestion.get(fn_id),
        )
        if fig_space:
            st.plotly_chart(fig_space, use_container_width=True, config={"displayModeBar": False})

    # ---------------------------------------------------------------
    # TAB 2: HISTORY CHART
    # ---------------------------------------------------------------
    with tabs[1]:
        fig = make_history_chart(fn_id, color, fn_h, initial_data=initial_data)
        if fig:
            st.markdown('<div class="section-label">Portal Submissions Over Time</div>', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            Y = fn_h["Y"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Best", f"{max(Y):.4f}")
            s2.metric("Latest", f"{Y[-1]:.4f}")
            s3.metric("Mean", f"{np.mean(Y):.4f}")
            s4.metric("Observations", len(Y))

            df_cols = {
                "Week": range(1, len(Y)+1),
                **{cfg["dim_labels"][i]: [x[i] for x in fn_h["X"]] for i in range(cfg["dims"])},
                "Output": [round(y, 6) for y in Y],
            }
            fn_meta = fn_h.get("meta", [])
            if any(fn_meta):
                df_cols["Acq"] = [
                    (fn_meta[i].get("acq","").upper() if i < len(fn_meta) else "") for i in range(len(Y))
                ]
                df_cols["β"] = [
                    (fn_meta[i].get("beta","") if i < len(fn_meta) else "") for i in range(len(Y))
                ]
                df_cols["ξ"] = [
                    (fn_meta[i].get("xi","") if i < len(fn_meta) else "") for i in range(len(Y))
                ]
                df_cols["Kernel"] = [
                    (fn_meta[i].get("kernel","") if i < len(fn_meta) else "") for i in range(len(Y))
                ]
            df = pd.DataFrame(df_cols)
            st.dataframe(
                df.style.highlight_max(subset=["Output"], color="#f59e0b22"),
                use_container_width=True, hide_index=True
            )
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                        border:1px dashed #1e2d45;border-radius:8px">
              No observations yet for this function.<br>Use the Query tab to record your first result.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-label">GP Posterior — Mean & 95% Confidence Interval</div>', unsafe_allow_html=True)
        st.caption("1D slices through the GP: all other dimensions held fixed at the best-known point.")
        fig_gp = make_gp_slice_plot(fn_id, color, history, initial_data,
                                    suggestion=st.session_state.suggestion.get(fn_id))
        if fig_gp:
            st.plotly_chart(fig_gp, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("""
            <div style="text-align:center;padding:1.5rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                        border:1px dashed #1e2d45;border-radius:8px">
              Need at least 2 data points to fit the GP.
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="section-label">Acquisition Function Comparison</div>', unsafe_allow_html=True)
        st.caption("All four acquisition functions normalised to [0, 1] — shows where each would rank as the next best query along each dimension.")
        _beta_viz = st.session_state.acq_overrides.get(fn_id) and cfg["beta"] or cfg["beta"]
        fig_acq = make_acq_comparison_plot(fn_id, history, initial_data, cfg["beta"], cfg["xi"])
        if fig_acq:
            st.plotly_chart(fig_acq, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown("""
            <div style="text-align:center;padding:1.5rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                        border:1px dashed #1e2d45;border-radius:8px">
              Need at least 2 data points to render acquisition comparison.
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # TAB 3: REFLECTION
    # ---------------------------------------------------------------
    with tabs[2]:
        reflection_text = generate_reflection(fn_id, history)
        st.markdown(f'<div class="reflection-box">{reflection_text}</div>', unsafe_allow_html=True)
        if fn_h["Y"]:
            st.markdown("")
            if st.button("📋  Copy to clipboard", key=f"copy_{fn_id}"):
                st.code(reflection_text, language=None)

    # ---------------------------------------------------------------
    # TAB 4: STRATEGY
    # ---------------------------------------------------------------
    with tabs[3]:
        st.markdown(f'<div class="strategy-box">{cfg["notes"]}</div>', unsafe_allow_html=True)

        if fn_id == 7:
            st.markdown("")
            st.markdown('<div class="section-label">GBM Parameter Guide</div>', unsafe_allow_html=True)

            gbm_data = {
                "Dimension": ["Dim 1","Dim 2","Dim 3","Dim 4","Dim 5","Dim 6"],
                "Parameter": ["n_estimators","learning_rate","max_depth","subsample","max_features","regularisation"],
                "Direction": ["0 = few, 1 = many","0 = small, 1 = large","0 = shallow, 1 = deep","0 = 50%, 1 = 100%","0 = few, 1 = all","0 = minimal, 1 = heavy"],
                "Best Zone": ["~0.33","~0.31","~0.25","~0.80","~0.80","~0.05"],
            }
            st.dataframe(pd.DataFrame(gbm_data), use_container_width=True, hide_index=True)
            st.markdown("""
            <div class="strategy-box" style="margin-top:0.75rem">
              <strong style="color:#e2e8f0">Key relationships:</strong><br>
              Dim 2 ↑ (learning_rate) → Dim 1 ↓ (n_estimators)<br>
              Dim 3 ↑ (max_depth) → Dim 4 ↓ (subsample)<br>
              Noisy output → Dim 6 ↑ (more regularisation)
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # TAB 5: AI ANALYSIS
    # ---------------------------------------------------------------
    with tabs[4]:
        st.markdown('<div class="section-label">AI-Powered Analysis & Strategy</div>', unsafe_allow_html=True)

        api_key = _get_api_key()
        if not api_key:
            st.markdown("""
            <div style="background:#0a0e1a;border:1px solid #f59e0b44;border-radius:8px;
                        padding:0.9rem 1rem;font-family:'JetBrains Mono',monospace;
                        font-size:0.78rem;color:#f59e0b">
              ⚠️ No Anthropic API key found.<br>
              Set <code>ANTHROPIC_API_KEY</code> in your environment or <code>.streamlit/secrets.toml</code>,
              or enter it below.
            </div>
            """, unsafe_allow_html=True)
            api_key = st.text_input(
                "Anthropic API Key", type="password",
                placeholder="sk-ant-...",
                key=f"api_key_input_{fn_id}",
                label_visibility="collapsed",
            )

        n_obs = len(fn_h["Y"])
        cache_key = (fn_id, n_obs)
        cached = st.session_state.ai_cache.get(fn_id)
        cached_valid = cached and cached.get("cache_key") == cache_key

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            run_analysis = st.button(
                "✨ Generate Analysis" if not cached_valid else "🔄 Refresh Analysis",
                key=f"ai_run_{fn_id}",
                type="primary", use_container_width=True,
                disabled=not api_key,
            )
        with col_btn2:
            if cached_valid:
                stored = fn_h.get("ai_analyses", [])
                latest = stored[-1] if stored else {}
                ts = latest.get("timestamp", "")
                ts_str = f" · saved {ts}" if ts else ""
                st.markdown(
                    f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.68rem;'
                    f'color:#64748b">Analysis for week {latest.get("week", "?")} '
                    f'({latest.get("obs_count", n_obs)} obs){ts_str}</span>',
                    unsafe_allow_html=True,
                )

        if run_analysis and api_key:
            with st.spinner("Calling Claude..."):
                result = ai_analysis(fn_id, history, initial_data, api_key)
            ts_now = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = {
                "week": fn_h.get("week", n_obs),
                "obs_count": n_obs,
                "timestamp": ts_now,
                "text": result,
            }
            if "ai_analyses" not in fn_h:
                fn_h["ai_analyses"] = []
            fn_h["ai_analyses"].append(entry)
            save_history(history)
            st.session_state.ai_cache[fn_id] = {"cache_key": cache_key, "text": result}
            cached_valid = True

        if cached_valid:
            text = st.session_state.ai_cache[fn_id]["text"]
            st.markdown(f'<div class="ai-box">{text}</div>', unsafe_allow_html=True)

            # Past analyses archive
            past = fn_h.get("ai_analyses", [])
            if len(past) > 1:
                st.markdown("")
                with st.expander(f"📚  Analysis history ({len(past)} saved)", expanded=False):
                    for entry in reversed(past):
                        st.markdown(
                            f'<div class="ai-label">Week {entry.get("week","?")} · '
                            f'{entry.get("obs_count","?")} obs · {entry.get("timestamp","")}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'<div class="ai-box" style="margin-bottom:0.75rem">{entry["text"]}</div>',
                                    unsafe_allow_html=True)
        elif not run_analysis:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                        border:1px dashed #1e2d45;border-radius:8px">
              Click "Generate Analysis" to have Claude review your observations<br>
              and recommend a strategy for the next query.
            </div>
            """, unsafe_allow_html=True)

    # ---------------------------------------------------------------
    # TAB 6: JOURNAL — per-week observations, strategy, reflection, AI
    # ---------------------------------------------------------------
    with tabs[5]:
        st.markdown('<div class="section-label">Weekly Journal</div>', unsafe_allow_html=True)

        if not fn_h["Y"]:
            st.markdown("""
            <div style="text-align:center;padding:3rem;color:#64748b;
                        font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                        border:1px dashed #1e2d45;border-radius:8px">
              No observations recorded yet for this function.<br>
              Record your first result in the Query tab to start the journal.
            </div>
            """, unsafe_allow_html=True)
        else:
            fn_meta = fn_h.get("meta", [])
            ai_analyses = fn_h.get("ai_analyses", [])

            # Index AI analyses by obs_count so we can look them up per week
            ai_by_obs: dict[int, list[dict]] = {}
            for _entry in ai_analyses:
                _oc = _entry.get("obs_count", 0)
                ai_by_obs.setdefault(_oc, []).append(_entry)

            init = (initial_data or {}).get(fn_id)
            init_best = float(init["Y"].max()) if init is not None else None

            # Render newest week first
            for i in range(len(fn_h["Y"]) - 1, -1, -1):
                week_num = i + 1
                x_vec = fn_h["X"][i]
                y_val = fn_h["Y"][i]
                mi = fn_meta[i] if i < len(fn_meta) else {}

                # Determine improvement vs previous portal best
                prev_best = max(fn_h["Y"][:i]) if i > 0 else None
                curr_best = max(fn_h["Y"][:i + 1])
                is_all_time_best = (y_val == max(fn_h["Y"]))
                improved = prev_best is None or y_val > prev_best

                # Card accent class
                if is_all_time_best:
                    card_cls = "week-card week-card-best"
                elif improved:
                    card_cls = "week-card week-card-improved"
                else:
                    card_cls = "week-card week-card-same"

                # Delta label
                if prev_best is None:
                    delta_html = '<span class="week-delta-new">First submission</span>'
                elif improved:
                    pct = (y_val - prev_best) / abs(prev_best) * 100 if prev_best != 0 else 0
                    delta_html = f'<span class="week-delta-up">▲ +{pct:.1f}% vs prev best</span>'
                else:
                    pct = (y_val - prev_best) / abs(prev_best) * 100 if prev_best != 0 else 0
                    delta_html = f'<span class="week-delta-down">▼ {pct:.1f}% vs prev best</span>'

                # AI badge if analysis exists for this week
                has_ai = week_num in ai_by_obs
                ai_badge = '<span class="week-ai-badge">AI</span>' if has_ai else ""

                # Acquisition settings
                acq_str = mi.get("acq", cfg["acquisition"]).upper()
                beta_str = mi.get("beta", cfg["beta"])
                xi_str   = mi.get("xi",   cfg["xi"])
                kernel_str = mi.get("kernel", cfg["kernel"])
                vs_init = ""
                if init_best is not None:
                    diff = y_val - init_best
                    sign = "+" if diff >= 0 else ""
                    vs_init = f'{sign}{diff:.4f} vs initial best'

                # Per-week mini reflection
                if prev_best is None:
                    refl = (
                        f"First portal submission for this function. "
                        f"Submitted input [{', '.join(f'{v:.4f}' for v in x_vec)}] "
                        f"and received Y = {y_val:.4f}. "
                    )
                    if init_best is not None:
                        if y_val > init_best:
                            refl += f"This exceeded the initial data best of {init_best:.4f} — a strong start."
                        else:
                            refl += (
                                f"The initial data best is {init_best:.4f}; this submission is below that. "
                                f"The GP needs more data to converge on the peak region."
                            )
                elif improved:
                    refl = (
                        f"Submitted [{', '.join(f'{v:.4f}' for v in x_vec)}] → Y = {y_val:.4f}. "
                        f"Improvement of {y_val - prev_best:+.4f} over previous best {prev_best:.4f}. "
                        f"The {acq_str} acquisition function (β={beta_str}, ξ={xi_str}) directed the search "
                        f"to a more promising region. Continue exploiting this neighbourhood."
                    )
                else:
                    refl = (
                        f"Submitted [{', '.join(f'{v:.4f}' for v in x_vec)}] → Y = {y_val:.4f}. "
                        f"Did not improve on the best of {curr_best:.4f}. "
                        f"The {acq_str} acquisition settings (β={beta_str}, ξ={xi_str}) may need adjustment — "
                        f"consider {'increasing β to explore more broadly' if acq_str == 'UCB' else 'increasing ξ to escape the current local region'}."
                    )

                y_val_cls = "week-cell-value week-cell-value-hi" if is_all_time_best else "week-cell-value"

                st.markdown(f"""
                <div class="{card_cls}">
                  <div class="week-header">
                    <span class="week-num">WEEK {week_num}{ai_badge}</span>
                    {delta_html}
                  </div>
                  <div class="week-grid">
                    <div class="week-cell">
                      <div class="week-cell-label">Input submitted</div>
                      <div class="week-cell-value">[{', '.join(f'{v:.4f}' for v in x_vec)}]</div>
                    </div>
                    <div class="week-cell">
                      <div class="week-cell-label">Output received</div>
                      <div class="{y_val_cls}">{y_val:.6g}{'  ★ all-time best' if is_all_time_best else ''}</div>
                    </div>
                    <div class="week-cell">
                      <div class="week-cell-label">Acquisition settings</div>
                      <div class="week-cell-value">{acq_str}  β={beta_str}  ξ={xi_str}  {kernel_str}</div>
                    </div>
                    <div class="week-cell">
                      <div class="week-cell-label">vs initial data best</div>
                      <div class="week-cell-value">{vs_init if vs_init else '—'}</div>
                    </div>
                  </div>
                  <div class="week-reflection">{refl}</div>
                </div>
                """, unsafe_allow_html=True)

                # AI analyses for this week (newest first)
                if has_ai:
                    week_analyses = sorted(ai_by_obs[week_num],
                                           key=lambda e: e.get("timestamp", ""), reverse=True)
                    for _a in week_analyses:
                        ts = _a.get("timestamp", "")
                        label = f"AI Analysis · Week {week_num} · {ts}"
                        with st.expander(label, expanded=(week_num == len(fn_h["Y"]))):
                            st.markdown(f'<div class="ai-box">{_a["text"]}</div>',
                                        unsafe_allow_html=True)

# =============================================================================
# BOTTOM: ALL-FUNCTIONS SUMMARY
# =============================================================================
st.markdown("---")
st.markdown('<div class="section-label">All Functions Summary</div>', unsafe_allow_html=True)

sum_left, sum_right = st.columns([2, 1], gap="large")

with sum_left:
    fig_sum = make_summary_chart(history)
    if fig_sum:
        st.plotly_chart(fig_sum, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#64748b;
                    font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                    border:1px dashed #1e2d45;border-radius:8px">
          No results yet — start submitting queries to see the comparison chart.
        </div>
        """, unsafe_allow_html=True)

with sum_right:
    rows = []
    for i in range(1, 9):
        fn_h_i = history[i]
        rows.append({
            "Fn": f"F{i}",
            "Dims": FUNCTION_CONFIG[i]["dims"],
            "Week": fn_h_i.get("week", 0),
            "Obs": len(fn_h_i["Y"]),
            "Best": round(max(fn_h_i["Y"]), 4) if fn_h_i["Y"] else None,
            "Acq": st.session_state.acq_overrides.get(i, FUNCTION_CONFIG[i]["acquisition"]).upper(),
        })
    df_sum = pd.DataFrame(rows)
    st.dataframe(df_sum, use_container_width=True, hide_index=True)

    # Reset button
    st.markdown("")
    with st.expander("⚠️ Danger Zone"):
        st.warning("This will delete ALL observation history.")
        if st.button("Reset All History", type="primary"):
            st.session_state.history = {i: {"X": [], "Y": [], "week": 0} for i in range(1, 9)}
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.session_state.suggestion = {}
            st.session_state.gp_info = {}
            st.rerun()
