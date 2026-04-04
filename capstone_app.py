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
import warnings
from datetime import datetime
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

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
  .pill-ucb { background: rgba(0,212,255,0.15); color: #00d4ff; }
  .pill-ei  { background: rgba(16,185,129,0.15); color: #10b981; }
  .pill-pi  { background: rgba(124,58,237,0.15); color: #a78bfa; }
  .pill-var { background: rgba(245,158,11,0.15); color: #f59e0b; }

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
    display: grid; grid-template-columns: 36px 1fr auto;
    gap: 0.5rem; padding: 0.3rem 0;
    border-bottom: 1px solid #1e2d4530;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
    align-items: center;
  }
  .obs-week { color: #64748b; }
  .obs-input { color: #cbd5e1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .obs-output { color: #10b981; font-weight: 600; text-align: right; }
  .obs-best { color: #f59e0b !important; }

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
        "kernel": "rbf", "acquisition": "ucb", "beta": 2.0, "xi": 0.05,
        "description": "2D contamination/radiation field",
        "dim_labels": ["x₁ (position)", "x₂ (position)"],
        "notes": "Sparse non-zero regions suggest the signal is localised. UCB with beta=2.0 keeps exploration high early to avoid missing the signal entirely. Reduce beta to 1.0 after a non-zero reading.",
    },
    2: {
        "dims": 2, "bounds": [(0.0,1.0)]*2,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.5, "xi": 0.1,
        "description": "2D noisy black-box log-likelihood",
        "dim_labels": ["x₁", "x₂"],
        "notes": "Explicitly noisy with local optima. Matern kernel handles rougher functions. High beta UCB resists premature exploitation of noisy early readings.",
    },
    3: {
        "dims": 3, "bounds": [(0.0,1.0)]*3,
        "kernel": "matern", "acquisition": "ei", "beta": 1.96, "xi": 0.02,
        "description": "3D drug compound combinations",
        "dim_labels": ["Compound A", "Compound B", "Compound C"],
        "notes": "Physical process — Matern kernel appropriate. EI balances improvement focus with uncertainty. Negated side effects = maximise.",
    },
    4: {
        "dims": 4, "bounds": [(0.0,1.0)]*4,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.0, "xi": 0.05,
        "description": "4D warehouse ML hyperparameter tuning",
        "dim_labels": ["Param 1", "Param 2", "Param 3", "Param 4"],
        "notes": "Rough landscape with local optima — stay exploratory early. Dynamic environment means old observations may drift in value.",
    },
    5: {
        "dims": 4, "bounds": [(0.0,1.0)]*4,
        "kernel": "rbf", "acquisition": "ei", "beta": 1.5, "xi": 0.01,
        "description": "4D chemical yield (unimodal)",
        "dim_labels": ["Chemical 1", "Chemical 2", "Chemical 3", "Chemical 4"],
        "notes": "Unimodal — safest to exploit aggressively once in the right neighbourhood. Low xi on EI. Shift to pure exploitation by week 4.",
    },
    6: {
        "dims": 5, "bounds": [(0.0,1.0)]*5,
        "kernel": "matern", "acquisition": "ei", "beta": 1.96, "xi": 0.02,
        "description": "5D cake recipe (negative penalty, max→0)",
        "dim_labels": ["Flour", "Sugar", "Eggs", "Butter", "Milk"],
        "notes": "Output is negative by design — scores near zero are best. Start near balanced ingredient ratios. Extreme values almost certainly score worse.",
    },
    7: {
        "dims": 6, "bounds": [(0.0,1.0)]*6,
        "kernel": "matern", "acquisition": "ei", "beta": 1.96, "xi": 0.05,
        "description": "6D gradient boosting hyperparameters",
        "dim_labels": [
            "Dim 1: n_estimators [0–1]", "Dim 2: learning_rate [0–1]",
            "Dim 3: max_depth [0–1]",    "Dim 4: subsample [0–1]",
            "Dim 5: max_features [0–1]", "Dim 6: regularisation [0–1]",
        ],
        "notes": "Almost certainly GBM — all inputs normalised to [0,1] by the platform. EI appropriate given structured landscape. Key: learning_rate (dim2) and n_estimators (dim1) are inversely related.",
        "informed_start": [0.333, 0.310, 0.250, 0.800, 0.800, 0.050],
    },
    8: {
        "dims": 8, "bounds": [(0.0,1.0)]*8,
        "kernel": "matern", "acquisition": "ucb", "beta": 2.5, "xi": 0.1,
        "description": "8D complex black-box (ML hyperparameters)",
        "dim_labels": [f"Param {i+1}" for i in range(8)],
        "notes": "Hardest function. GP at 8D will be uncertain — accept this. High beta UCB keeps exploration broad. Focus on process quality in reflections rather than absolute score.",
    },
}

# =============================================================================
# PERSISTENCE
# =============================================================================
HISTORY_FILE = "capstone_history.json"


def align_xy_pair(fn_h: dict) -> bool:
    """
    Ensure X and Y have the same length (paired observations).
    Truncates the longer list. Returns True if anything was changed.
    """
    X, Y = fn_h.get("X", []), fn_h.get("Y", [])
    nx, ny = len(X), len(Y)
    if nx == ny:
        return False
    n = min(nx, ny)
    fn_h["X"] = X[:n]
    fn_h["Y"] = Y[:n]
    w = fn_h.get("week", 0)
    if w > n:
        fn_h["week"] = n
    return True


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            raw = json.load(f)
        history = {int(k): v for k, v in raw.items()}
        dirty = False
        for _fid, fn_h in history.items():
            if align_xy_pair(fn_h):
                dirty = True
        if dirty:
            try:
                with open(HISTORY_FILE, "w") as f:
                    json.dump(history, f, indent=2)
            except OSError:
                pass
        return history
    return {i: {"X": [], "Y": [], "week": 0} for i in range(1, 9)}

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

# =============================================================================
# GP + ACQUISITION
# =============================================================================
def build_gp(kernel_type="matern"):
    if kernel_type == "rbf":
        kernel = C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
    else:
        kernel = C(1.0) * Matern(length_scale=1.0, nu=2.5, length_scale_bounds=(1e-2, 10.0))
    return GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)

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
    return mean

def suggest_next(fn_id, history, acq_override=None, beta_override=None, xi_override=None):
    cfg = FUNCTION_CONFIG[fn_id]
    fn_h = history[fn_id]
    acq  = acq_override  or cfg["acquisition"]
    beta = beta_override or cfg["beta"]
    xi   = xi_override   or cfg["xi"]

    if not fn_h["X"]:
        if "informed_start" in cfg:
            return cfg["informed_start"], None, None
        return [round(np.random.uniform(0,1), 6) for _ in range(cfg["dims"])], None, None

    if align_xy_pair(fn_h):
        try:
            save_history(history)
        except OSError:
            pass
    X = np.asarray(fn_h["X"], dtype=np.float64)
    Y = np.asarray(fn_h["Y"], dtype=np.float64)
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    if len(Y) == 0:
        if "informed_start" in cfg:
            return cfg["informed_start"], None, None
        return [round(np.random.uniform(0, 1), 6) for _ in range(cfg["dims"])], None, None

    if not np.all(np.isfinite(Y)):
        Y = np.nan_to_num(Y, nan=0.0, posinf=1e300, neginf=-1e300)
    # Extreme magnitudes (e.g. typo 1e-185 or huge scores) destabilise the GP kernel matrix.
    Y = np.clip(Y, -1e12, 1e12)

    gp = build_gp(cfg["kernel"])
    gp.fit(X, Y)

    n_cand = 5000
    candidates = np.random.uniform(0, 1, (n_cand, cfg["dims"]))
    mean, std = gp.predict(candidates, return_std=True)
    y_max = Y.max()
    scores = compute_acquisition(acq, mean, std, y_max, beta, xi)
    best = np.argmax(scores)
    suggestion = np.clip(candidates[best], 0.0, 1.0)
    return [round(float(v), 6) for v in suggestion], float(mean[best]), float(std[best])

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

def make_history_chart(fn_id, color, fn_h):
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
        name="Output",
    ))
    fig.add_hline(y=y_max, line_dash="dash", line_color="#f59e0b", line_width=1,
                  annotation_text=f"Best: {y_max:.4f}", annotation_font_color="#f59e0b",
                  annotation_font_size=10)
    fig.update_layout(**PLOTLY_LAYOUT, height=220,
                      xaxis_title="Week", yaxis_title="Output",
                      showlegend=False)
    fig.update_xaxes(tickvals=weeks, dtick=1)
    return fig

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

history = st.session_state.history

# Auto-generate week-1 suggestions for any function that has no observations yet
# and hasn't had a suggestion generated yet — so inputs are never blank on load
for _fn_id in range(1, 9):
    if _fn_id not in st.session_state.suggestion:
        _sug, _, _ = suggest_next(_fn_id, history)
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
        pill_cls = f"pill-{acq.lower()}" if acq.lower() in ("ucb","ei","pi") else "pill-var"

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

    tabs = st.tabs(["📥  Query", "📊  History", "📝  Reflection", "⚙️  Strategy"])

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
            acq_choice = st.selectbox(
                "Method", ["ucb", "ei", "pi", "variance"],
                index=["ucb","ei","pi","variance"].index(
                    st.session_state.acq_overrides.get(fn_id, cfg["acquisition"])
                ),
                key=f"acq_{fn_id}",
                label_visibility="collapsed"
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
                    save_history(history)
                    suggestion, gp_mean, gp_std = suggest_next(
                        fn_id, history,
                        acq_override=acq_choice,
                        beta_override=beta_val,
                        xi_override=xi_val,
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
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem">
                      <div style="background:#0a0e1a;border:1px solid #1e2d45;border-radius:6px;padding:0.4rem;text-align:center">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#00d4ff;font-weight:600">{m:.4f}</div>
                        <div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;letter-spacing:0.06em">GP Mean</div>
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
                rows_html = ""
                for i, (x, y) in enumerate(zip(fn_h["X"], fn_h["Y"])):
                    is_best = y == y_max_val
                    out_cls = "obs-best" if is_best else "obs-output"
                    star = " ★" if is_best else ""
                    rows_html += f"""
                    <div class="obs-row">
                      <div class="obs-week">W{i+1}</div>
                      <div class="obs-input">[{', '.join(f'{v:.3f}' for v in x)}]</div>
                      <div class="{out_cls}">{y:.4f}{star}</div>
                    </div>"""
                st.markdown(f'<div style="max-height:180px;overflow-y:auto">{rows_html}</div>', unsafe_allow_html=True)

                # Delete last observation
                if st.button("↩ Undo last observation", key=f"undo_{fn_id}"):
                    fn_h["X"].pop(); fn_h["Y"].pop()
                    fn_h["week"] = max(0, fn_h.get("week", 1) - 1)
                    save_history(history)
                    st.rerun()

    # ---------------------------------------------------------------
    # TAB 2: HISTORY CHART
    # ---------------------------------------------------------------
    with tabs[1]:
        fig = make_history_chart(fn_id, color, fn_h)
        if fig:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            # Stats row
            Y = fn_h["Y"]
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Best", f"{max(Y):.4f}")
            s2.metric("Latest", f"{Y[-1]:.4f}")
            s3.metric("Mean", f"{np.mean(Y):.4f}")
            s4.metric("Observations", len(Y))

            # Data table
            df = pd.DataFrame({
                "Week": range(1, len(Y)+1),
                **{cfg["dim_labels"][i]: [x[i] for x in fn_h["X"]] for i in range(cfg["dims"])},
                "Output": [round(y, 6) for y in Y],
            })
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
