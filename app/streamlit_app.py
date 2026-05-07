from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


st.set_page_config(
    page_title="Formulation Bayesian Optimization",
    page_icon="💊",
    layout="wide",
)


API_PCT = 30.0
TOTAL_EXCIPIENT_PCT = 100.0 - API_PCT
BOUNDS: Dict[str, Tuple[float, float]] = {
    "hpmc": (0.0, 20.0),
    "mcc": (20.0, 60.0),
    "ccs": (1.0, 8.0),
    "mgst": (0.25, 2.0),
    "pvp": (0.0, 10.0),
}
FORMULATION_COLS = ["hpmc", "mcc", "ccs", "mgst", "pvp"]
DISPLAY_NAMES = {
    "hpmc": "HPMC (Hydroxypropyl Methylcellulose)",
    "mcc": "MCC (Microcrystalline Cellulose)",
    "ccs": "CCS (Croscarmellose Sodium)",
    "mgst": "MgSt (Magnesium Stearate)",
    "pvp": "PVP K30 (Polyvinylpyrrolidone)",
}
Q45_HEATMAP_SCALE = [
    [0.0, "rgb(245, 247, 249)"],
    [0.25, "rgb(215, 230, 240)"],
    [0.5, "rgb(169, 203, 224)"],
    [0.75, "rgb(111, 160, 196)"],
    [1.0, "rgb(63, 115, 157)"],
]
UNC_HEATMAP_SCALE = [
    [0.0, "rgb(248, 250, 252)"],
    [0.35, "rgb(226, 232, 240)"],
    [0.7, "rgb(148, 163, 184)"],
    [1.0, "rgb(100, 116, 139)"],
]

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@1&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

:root {
    --bg:           #f7f0e8;
    --ink:          #193a3b;
    --muted:        rgba(25,58,59,0.55);
    --teal:         #193a3b;
    --teal-light:   #2a5f61;
    --gold:         #c9a84c;
    --gold-light:   #e8c06a;
    --gold-muted:   rgba(201,168,76,0.12);
    --card:         rgba(255,255,255,0.52);
    --card-solid:   #ffffff;
    --line:         rgba(25,58,59,0.08);
    --glass-border: rgba(255,255,255,0.72);
    --shadow-card:  0 4px 24px -1px rgba(25,58,59,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.85),
                    inset 0 -1px 0 rgba(255,255,255,0.22);
    --shadow-chart: 0 10px 40px -10px rgba(25,58,59,0.09);
}

/* ── Fonts ─────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    font-size: 18px;
}

/* ── Canvas ─────────────────────────────── */
.stApp, .main .block-container {
    background: var(--bg) !important;
}
.main .block-container {
    padding-top: 2rem !important;
    max-width: 1280px;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Scrollbar ──────────────────────────── */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(25,58,59,0.14); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(25,58,59,0.28); }

/* ── Sidebar ────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(175deg, #193a3b 0%, #1e4647 55%, #224c4d 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
    box-shadow: 4px 0 24px rgba(0,0,0,0.12);
}
[data-testid="stSidebar"] * { color: rgba(236,247,245,0.9) !important; }
[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

/* Sidebar nav items */
[data-testid="stSidebar"] .stRadio label {
    display: flex;
    align-items: center;
    border-radius: 10px;
    padding: 0.5rem 0.7rem !important;
    font-size: 1.2rem;
    font-weight: 500;
    transition: background 0.18s ease, color 0.18s ease;
    position: relative;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #fff !important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
[data-testid="stSidebar"] .stRadio input:checked + label {
    background: rgba(42,95,97,0.45) !important;
    color: var(--gold) !important;
    border-left: 3px solid var(--gold);
    padding-left: calc(0.7rem - 3px) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.1) !important;
    margin: 0.6rem 0;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] small {
    color: rgba(236,247,245,0.5) !important;
    font-size: 1.08rem !important;
}

[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: rgba(236,247,245,0.95) !important;
}

[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
    font-size: 1.2rem !important;
    line-height: 1.4 !important;
}

/* ── Metric cards — glassmorphism ───────── */
[data-testid="stMetric"] {
    background: var(--card) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 18px !important;
    box-shadow: var(--shadow-card) !important;
    padding: 1rem 1.1rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px -4px rgba(25,58,59,0.12),
                inset 0 1px 0 rgba(255,255,255,0.9) !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    color: var(--teal) !important;
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums !important;
    letter-spacing: -0.04em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
}

/* ── Buttons ────────────────────────────── */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
    border: 1px solid rgba(25,58,59,0.18) !important;
    transition: all 0.18s ease !important;
    font-size: 0.96rem !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold) 0%, var(--gold-light) 100%) !important;
    border: none !important;
    color: #152e2f !important;
    box-shadow: 0 8px 24px -4px rgba(201,168,76,0.38) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 12px 32px -4px rgba(201,168,76,0.55) !important;
    transform: translateY(-2px) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: var(--gold-muted) !important;
    border-color: var(--gold) !important;
    color: #193a3b !important;
}

/* ── Inputs ─────────────────────────────── */
.stNumberInput input,
.stTextInput input,
.stSelectbox div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: rgba(25,58,59,0.16) !important;
    background: rgba(255,255,255,0.72) !important;
}
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.18) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--teal) !important;
    border-color: var(--gold) !important;
}

/* ── Dataframe / Table ──────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-chart) !important;
    border: 1px solid var(--line) !important;
}
[data-testid="stDataFrame"] thead th {
    background: rgba(247,240,232,0.5) !important;
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.09em !important;
    color: var(--muted) !important;
    border-bottom: 1px solid var(--line) !important;
}

/* ── Charts (Plotly iframes) ────────────── */
.stPlotlyChart {
    border-radius: 20px !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-chart) !important;
    background: var(--card-solid) !important;
    border: 1px solid rgba(25,58,59,0.06) !important;
}

/* ── Alerts ─────────────────────────────── */
.stAlert {
    border-radius: 14px !important;
    border: 1px solid var(--line) !important;
    font-size: 0.875rem !important;
}
.stAlert[data-baseweb="notification"][kind="success"] {
    background: rgba(209,250,229,0.45) !important;
    border-color: #a7f3d0 !important;
}
.stAlert[data-baseweb="notification"][kind="error"] {
    background: rgba(254,226,226,0.45) !important;
}

/* ── Forms ──────────────────────────────── */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.38) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    padding: 1.4rem !important;
    box-shadow: var(--shadow-card) !important;
}

/* ── Spinner ────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--gold) !important;
}

/* ── Typography ─────────────────────────── */
h1, h2, h3, h4 {
    color: var(--teal) !important;
    letter-spacing: -0.025em;
}
h1 { font-size: 2.2rem !important; }
h2 { font-size: 1.6rem !important; }
h3 { font-size: 1.28rem !important; }

h1 em {
    font-family: 'Instrument Serif', Georgia, serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    color: var(--teal-light) !important;
}

/* subheader gold underline accent */
h2::after {
    content: '';
    display: block;
    width: 2.4rem;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), var(--gold-light));
    border-radius: 99px;
    margin-top: 6px;
}

p, .stMarkdown p {
    color: rgba(25,58,59,0.8) !important;
    line-height: 1.65 !important;
    font-size: 1.04rem !important;
}

/* caption */
.stCaption p, [data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-size: 0.92rem !important;
}

/* ── JSON block ─────────────────────────── */
[data-testid="stJson"] {
    border-radius: 12px !important;
    background: rgba(247,240,232,0.6) !important;
    border: 1px solid var(--line) !important;
}

/* ── Sidebar brand header ───────────────── */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.2rem 0 1.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0.8rem;
}
.sidebar-brand-icon {
    width: 44px; height: 44px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
}
.sidebar-brand-name { font-size: 1.02rem; font-weight: 700; color: rgba(236,247,245,0.92); letter-spacing: 0.03em; }
.sidebar-brand-sub  { font-size: 0.82rem; font-weight: 600; color: #c9a84c; letter-spacing: 0.14em; text-transform: uppercase; }
</style>
"""


def apply_gnnbind_theme() -> None:
    """Apply the same Streamlit styling approach used in gnn-bind-optimizer."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def physics_q45(hpmc: np.ndarray, mcc: np.ndarray, ccs: np.ndarray, mgst: np.ndarray, pvp: np.ndarray) -> np.ndarray:
    """Physics-informed Q45 response surface."""
    ccs_gain = 24.0 * (1.0 - np.exp(-ccs / 2.2))
    q45 = (
        48.0
        + 0.45 * mcc
        + 0.90 * pvp
        + ccs_gain
        - 1.05 * hpmc
        - 0.035 * (hpmc**2)
        - 7.2 * mgst
        - 0.16 * hpmc * ccs
    )
    return np.clip(q45, 0.0, 100.0)


def simulate_dissolution(df: pd.DataFrame, noise_std: float = 2.0, seed: int = 0) -> np.ndarray:
    """Simulate observed Q45 (% API released at 45 min)."""
    rng = np.random.default_rng(seed)
    y = physics_q45(
        df["hpmc"].to_numpy(),
        df["mcc"].to_numpy(),
        df["ccs"].to_numpy(),
        df["mgst"].to_numpy(),
        df["pvp"].to_numpy(),
    )
    y = y + rng.normal(0.0, noise_std, size=len(df))
    return np.clip(y, 0.0, 100.0)


def is_feasible(df: pd.DataFrame) -> np.ndarray:
    checks = []
    for name, (lo, hi) in BOUNDS.items():
        checks.append(df[name].between(lo, hi))
    checks.append(np.isclose(df[FORMULATION_COLS].sum(axis=1), TOTAL_EXCIPIENT_PCT, atol=1e-6))
    return np.logical_and.reduce(checks)


def sample_feasible(n: int, seed: int = 0) -> pd.DataFrame:
    """Rejection sample feasible points with pvp as mass-balance remainder."""
    rng = np.random.default_rng(seed)
    rows: List[Tuple[float, float, float, float, float]] = []
    while len(rows) < n:
        hpmc = rng.uniform(*BOUNDS["hpmc"])
        mcc = rng.uniform(*BOUNDS["mcc"])
        ccs = rng.uniform(*BOUNDS["ccs"])
        mgst = rng.uniform(*BOUNDS["mgst"])
        pvp = TOTAL_EXCIPIENT_PCT - (hpmc + mcc + ccs + mgst)
        if BOUNDS["pvp"][0] <= pvp <= BOUNDS["pvp"][1]:
            rows.append((hpmc, mcc, ccs, mgst, pvp))
    return pd.DataFrame(rows, columns=FORMULATION_COLS)


def build_gp(df: pd.DataFrame) -> GaussianProcessRegressor:
    """Fit GP surrogate on observed experiments."""
    x = df[["hpmc", "mcc", "ccs", "mgst"]].to_numpy()
    y = df["q45"].to_numpy()
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(4), nu=2.5, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        random_state=0,
        n_restarts_optimizer=2,
    )
    gp.fit(x, y)
    return gp


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    sigma = np.clip(sigma, 1e-8, None)
    z = (mu - best) / sigma
    return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)


@dataclass
class Suggestion:
    point: Dict[str, float]
    mu: float
    sigma: float
    ei: float


def suggest_next(df: pd.DataFrame, seed: int = 0, n_candidates: int = 3000) -> Suggestion:
    gp = build_gp(df)
    candidates = sample_feasible(n_candidates, seed=seed)
    x = candidates[["hpmc", "mcc", "ccs", "mgst"]].to_numpy()
    mu, sigma = gp.predict(x, return_std=True)
    best = float(df["q45"].max())
    ei = expected_improvement(mu, sigma, best=best)

    # Avoid near-duplicates to keep exploration realistic.
    explored = df[["hpmc", "mcc", "ccs", "mgst"]].to_numpy()
    dists = np.sqrt(((x[:, None, :] - explored[None, :, :]) ** 2).sum(axis=2))
    min_dist = dists.min(axis=1)
    ei[min_dist < 0.15] = -np.inf

    idx = int(np.argmax(ei))
    row = candidates.iloc[idx].to_dict()
    return Suggestion(
        point={k: float(v) for k, v in row.items()},
        mu=float(mu[idx]),
        sigma=float(sigma[idx]),
        ei=float(ei[idx]),
    )


def create_initial_experiments(n_init: int = 10, seed: int = 42) -> pd.DataFrame:
    df = sample_feasible(n_init, seed=seed)
    df["q45"] = simulate_dissolution(df, seed=seed)
    df["source"] = "initial_design"
    df["timestamp"] = datetime.utcnow().isoformat()
    return df


def add_experiment(
    df: pd.DataFrame,
    row: Dict[str, float],
    source: str,
    observed_q45: float | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    new = pd.DataFrame([row])[FORMULATION_COLS]
    if observed_q45 is None:
        observed_q45 = float(simulate_dissolution(new, seed=seed)[0])
    new["q45"] = observed_q45
    new["source"] = source
    new["timestamp"] = datetime.utcnow().isoformat()
    return pd.concat([df, new], ignore_index=True)


def save_run(df: pd.DataFrame) -> Path:
    out_dir = Path(__file__).resolve().parents[1] / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    path = out_dir / f"{run_id}.json"
    payload = {"created_utc": datetime.utcnow().isoformat(), "n_experiments": int(len(df)), "records": df.to_dict(orient="records")}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@st.cache_data(show_spinner=False)
def benchmark_bo_vs_random(n_trials: int, n_eval: int, seed: int) -> pd.DataFrame:
    traces = []
    for t in range(n_trials):
        trial_seed = seed + t

        # Random baseline
        rnd = sample_feasible(n_eval, seed=trial_seed)
        rnd["q45"] = simulate_dissolution(rnd, seed=trial_seed)
        rnd_trace = np.maximum.accumulate(rnd["q45"].to_numpy())
        traces.append(pd.DataFrame({"evaluation": np.arange(1, n_eval + 1), "best_q45": rnd_trace, "method": "Random"}))

        # BO loop (sklearn GP + EI over random feasible candidates)
        bo = sample_feasible(10, seed=trial_seed + 100)
        bo["q45"] = simulate_dissolution(bo, seed=trial_seed + 100)
        best_trace = list(np.maximum.accumulate(bo["q45"].to_numpy()))
        for step in range(10, n_eval):
            suggestion = suggest_next(bo, seed=trial_seed + 500 + step, n_candidates=2500)
            bo = add_experiment(
                bo,
                suggestion.point,
                source="ai_suggested",
                observed_q45=None,
                seed=trial_seed + 2000 + step,
            )
            best_trace.append(max(best_trace[-1], float(bo.iloc[-1]["q45"])))
        traces.append(
            pd.DataFrame(
                {
                    "evaluation": np.arange(1, n_eval + 1),
                    "best_q45": np.array(best_trace[:n_eval]),
                    "method": "Bayesian Optimization",
                }
            )
        )

    out = pd.concat(traces, ignore_index=True)
    summary = (
        out.groupby(["method", "evaluation"], as_index=False)
        .agg(mean_best_q45=("best_q45", "mean"), std_best_q45=("best_q45", "std"))
        .fillna(0.0)
    )
    summary["ci95"] = 1.96 * summary["std_best_q45"] / np.sqrt(n_trials)
    return summary


def initialize_state() -> None:
    if "experiments" not in st.session_state:
        st.session_state.experiments = create_initial_experiments(n_init=10, seed=42)
    if "seed_counter" not in st.session_state:
        st.session_state.seed_counter = 1000
    if "latest_suggestion" not in st.session_state:
        st.session_state.latest_suggestion = None


def render_header(df: pd.DataFrame) -> None:
    best = df.sort_values("q45", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments Run", len(df))
    c2.metric("Best Q45 (%)", f"{best['q45']:.2f}")
    c3.metric("Target 80% Reached", "Yes" if best["q45"] >= 80 else "No")
    c4.metric("Target 85% Reached", "Yes" if best["q45"] >= 85 else "No")


def design_space_explorer(df: pd.DataFrame) -> None:
    st.subheader("Design Space Explorer")
    st.caption("Heatmap of GP-predicted dissolution and model uncertainty for chosen excipient pairs.")
    with st.expander("How this heatmap is generated", expanded=True):
        st.markdown(
            "- The model is fit on the **current experiment table** (initial + AI-suggested + manual rows).\n"
            "- A Gaussian Process (GP) surrogate is trained on observed `q45` values.\n"
            "- We then build a dense 2D grid over the selected X/Y excipients, keep other factors fixed, enforce mass-balance feasibility, and predict on each feasible grid point.\n"
            "- The left heatmap is GP posterior mean (`predicted Q45`), and the right heatmap is GP posterior standard deviation (`uncertainty`)."
        )
        st.markdown("**Simulator equation used to generate observed training Q45:**")
        st.latex(
            r"Q45 = 48 + 0.45\,MCC + 0.90\,PVP + 24\left(1-e^{-CCS/2.2}\right) - 1.05\,HPMC - 0.035\,HPMC^2 - 7.2\,MgSt - 0.16\,(HPMC\cdot CCS) + \epsilon,\ \epsilon\sim\mathcal{N}(0,2^2)"
        )
        st.markdown("**GP prediction shown on heatmap:**")
        st.latex(r"\hat{Q45}(x)=\mu_{GP}(x),\quad \text{uncertainty}(x)=\sigma_{GP}(x)")
    gp = build_gp(df)

    axis_options = ["hpmc", "mcc", "ccs", "mgst"]
    left, right = st.columns(2)
    x_axis = left.selectbox(
        "X-axis excipient",
        axis_options,
        index=0,
        format_func=lambda k: DISPLAY_NAMES[k],
    )
    y_axis = right.selectbox(
        "Y-axis excipient",
        [a for a in axis_options if a != x_axis],
        index=1,
        format_func=lambda k: DISPLAY_NAMES[k],
    )

    fixed = {}
    controls = [a for a in axis_options if a not in (x_axis, y_axis)]
    cols = st.columns(2)
    for i, name in enumerate(controls):
        lo, hi = BOUNDS[name]
        fixed[name] = cols[i].slider(f"Fixed {name.upper()} (% w/w)", float(lo), float(hi), float(df[name].median()), step=0.1)

    x_vals = np.linspace(*BOUNDS[x_axis], 70)
    y_vals = np.linspace(*BOUNDS[y_axis], 70)
    rows = []
    for y in y_vals:
        for x in x_vals:
            row = {"hpmc": np.nan, "mcc": np.nan, "ccs": np.nan, "mgst": np.nan}
            row[x_axis] = x
            row[y_axis] = y
            for k, v in fixed.items():
                row[k] = v
            pvp = TOTAL_EXCIPIENT_PCT - (row["hpmc"] + row["mcc"] + row["ccs"] + row["mgst"])
            if BOUNDS["pvp"][0] <= pvp <= BOUNDS["pvp"][1]:
                rows.append({**row, "pvp": pvp})

    grid = pd.DataFrame(rows)
    mu, sigma = gp.predict(grid[["hpmc", "mcc", "ccs", "mgst"]], return_std=True)
    grid["pred_q45"] = mu
    grid["pred_std"] = sigma

    _chart_layout = dict(
        paper_bgcolor="rgb(15, 23, 42)",
        plot_bgcolor="rgb(15, 23, 42)",
        font=dict(family="Space Grotesk, sans-serif", color="rgb(226, 232, 240)"),
        title_font=dict(size=14, weight="bold"),
        margin=dict(l=16, r=16, t=48, b=16),
        coloraxis_colorbar=dict(
            thickness=12,
            len=0.8,
            tickfont=dict(size=10, color="rgb(226, 232, 240)"),
            title_font=dict(size=10, color="rgb(226, 232, 240)"),
            outlinewidth=0,
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,0.18)",
            linecolor="rgba(148,163,184,0.32)",
            tickfont=dict(color="rgb(226, 232, 240)"),
            title=dict(font=dict(color="rgb(226, 232, 240)")),
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,0.18)",
            linecolor="rgba(148,163,184,0.32)",
            tickfont=dict(color="rgb(226, 232, 240)"),
            title=dict(font=dict(color="rgb(226, 232, 240)")),
        ),
    )

    p1 = px.density_heatmap(
        grid,
        x=x_axis,
        y=y_axis,
        z="pred_q45",
        histfunc="avg",
        nbinsx=45,
        nbinsy=45,
        color_continuous_scale=Q45_HEATMAP_SCALE,
        title="Predicted Q45 (%)",
        labels={x_axis: DISPLAY_NAMES[x_axis], y_axis: DISPLAY_NAMES[y_axis]},
    )
    p1.update_layout(**_chart_layout)
    p1.update_traces(colorscale=Q45_HEATMAP_SCALE)

    p2 = px.density_heatmap(
        grid,
        x=x_axis,
        y=y_axis,
        z="pred_std",
        histfunc="avg",
        nbinsx=45,
        nbinsy=45,
        color_continuous_scale=UNC_HEATMAP_SCALE,
        title="Prediction Uncertainty (Std Dev)",
        labels={x_axis: DISPLAY_NAMES[x_axis], y_axis: DISPLAY_NAMES[y_axis]},
    )
    p2.update_layout(**_chart_layout)
    p2.update_traces(colorscale=UNC_HEATMAP_SCALE)

    c1, c2 = st.columns(2)
    c1.plotly_chart(p1, use_container_width=True)
    c2.plotly_chart(p2, use_container_width=True)


def run_bo_page(df: pd.DataFrame) -> None:
    st.subheader("Run Bayesian Optimization")
    st.caption("Generate a suggested next experiment using expected improvement.")

    if st.button("Suggest Next Experiment (Run One BO Step)", type="primary"):
        suggestion = suggest_next(df, seed=st.session_state.seed_counter, n_candidates=3500)
        st.session_state.seed_counter += 1
        df = add_experiment(
            df,
            row=suggestion.point,
            source="ai_suggested",
            observed_q45=None,
            seed=st.session_state.seed_counter,
        )
        st.session_state.seed_counter += 1
        st.session_state.experiments = df
        st.session_state.latest_suggestion = suggestion

    if st.session_state.latest_suggestion is not None:
        s: Suggestion = st.session_state.latest_suggestion
        st.success("Suggested experiment executed and logged.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Q45", f"{s.mu:.2f}%")
        c2.metric("Uncertainty (Std)", f"{s.sigma:.2f}")
        c3.metric("Expected Improvement", f"{s.ei:.3f}")
        st.write(
            "Rationale: this point has strong expected gain while still sampling a region where model uncertainty is informative."
        )
        suggest_df = (
            pd.DataFrame(
                {
                    "Excipient": [k.upper() for k in FORMULATION_COLS],
                    "% w/w": [round(s.point[k], 3) for k in FORMULATION_COLS],
                }
            )
        )
        st.markdown("**Suggested formulation (% w/w):**")
        st.dataframe(suggest_df, use_container_width=True, hide_index=True)

    if st.button("Save Current Run to results/runs"):
        path = save_run(st.session_state.experiments)
        st.info(f"Saved run: {path}")


def experiment_log_page(df: pd.DataFrame) -> None:
    st.subheader("Experiment Log")
    st.caption("All formulations are shown in % w/w, with outcomes and provenance (AI-suggested vs manual).")
    st.dataframe(df.round(4), use_container_width=True, height=380)

    st.markdown("### Add Manual Experiment")
    with st.form("manual_experiment_form"):
        c1, c2, c3, c4 = st.columns(4)
        hpmc = c1.number_input("HPMC", min_value=BOUNDS["hpmc"][0], max_value=BOUNDS["hpmc"][1], value=8.0, step=0.1)
        mcc = c2.number_input("MCC", min_value=BOUNDS["mcc"][0], max_value=BOUNDS["mcc"][1], value=42.0, step=0.1)
        ccs = c3.number_input("CCS", min_value=BOUNDS["ccs"][0], max_value=BOUNDS["ccs"][1], value=4.0, step=0.1)
        mgst = c4.number_input("MgSt", min_value=BOUNDS["mgst"][0], max_value=BOUNDS["mgst"][1], value=1.0, step=0.05)
        pvp = TOTAL_EXCIPIENT_PCT - (hpmc + mcc + ccs + mgst)
        st.write(f"Derived PVP from mass balance: **{pvp:.3f}%**")
        q45_input = st.text_input("Observed Q45 (%) - leave blank to simulate", "")
        submitted = st.form_submit_button("Add Manual Experiment")

    if submitted:
        row = {"hpmc": hpmc, "mcc": mcc, "ccs": ccs, "mgst": mgst, "pvp": pvp}
        row_df = pd.DataFrame([row])
        if not bool(is_feasible(row_df)[0]):
            st.error("Formulation is infeasible under bounds/mass-balance. Adjust excipient percentages.")
            return

        observed = float(q45_input) if q45_input.strip() else None
        st.session_state.experiments = add_experiment(
            st.session_state.experiments,
            row=row,
            source="manual_entry",
            observed_q45=observed,
            seed=st.session_state.seed_counter,
        )
        st.session_state.seed_counter += 1
        st.success("Manual experiment added.")


_PREMIUM_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(247,240,232,0.35)",
    font=dict(family="Space Grotesk, sans-serif", color="#193a3b", size=12),
    title_font=dict(size=15, color="#193a3b"),
    margin=dict(l=16, r=16, t=52, b=16),
    legend=dict(
        bgcolor="rgba(255,255,255,0.6)",
        bordercolor="rgba(25,58,59,0.1)",
        borderwidth=1,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor="rgba(25,58,59,0.06)",
        linecolor="rgba(25,58,59,0.14)",
        zerolinecolor="rgba(25,58,59,0.08)",
    ),
    yaxis=dict(
        gridcolor="rgba(25,58,59,0.06)",
        linecolor="rgba(25,58,59,0.14)",
        zerolinecolor="rgba(25,58,59,0.08)",
    ),
)


def convergence_page(df: pd.DataFrame) -> None:
    st.subheader("Convergence")
    trace = np.maximum.accumulate(df["q45"].to_numpy())
    chart_df = pd.DataFrame({"Evaluation": np.arange(1, len(trace) + 1), "Best Q45": trace})
    fig = px.line(
        chart_df, x="Evaluation", y="Best Q45", markers=True,
        title="Best Observed Q45 vs Evaluation",
        color_discrete_sequence=["#193a3b"],
    )
    fig.update_traces(
        line=dict(width=2.5),
        marker=dict(size=7, color="#c9a84c", line=dict(color="#193a3b", width=1.5)),
    )
    fig.add_hline(y=80, line_dash="dash", line_color="#2a5f61", line_width=1.5,
                  annotation_text="Target 80%", annotation_font_color="#2a5f61")
    fig.add_hline(y=85, line_dash="dot", line_color="#c9a84c", line_width=1.5,
                  annotation_text="Stretch 85%", annotation_font_color="#c9a84c")
    fig.update_layout(**_PREMIUM_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def compare_page() -> None:
    st.subheader("Compare: BO vs Random Search")
    c1, c2, c3 = st.columns(3)
    n_trials = c1.slider("Trials", min_value=3, max_value=15, value=5, step=1)
    n_eval = c2.slider("Evaluations per trial", min_value=20, max_value=60, value=40, step=5)
    seed = c3.number_input("Random seed", min_value=0, max_value=100000, value=123, step=1)

    with st.spinner("Running benchmark..."):
        summary = benchmark_bo_vs_random(n_trials=n_trials, n_eval=n_eval, seed=int(seed))

    palette = {"Bayesian Optimization": "#193a3b", "Random": "#c9a84c"}
    ci_fill  = {"Bayesian Optimization": "rgba(25,58,59,0.10)", "Random": "rgba(201,168,76,0.12)"}

    fig = px.line(
        summary,
        x="evaluation",
        y="mean_best_q45",
        color="method",
        title="Mean Best Q45 vs Evaluation Budget",
        color_discrete_map=palette,
    )
    fig.update_traces(line=dict(width=2.5))
    for method in summary["method"].unique():
        sub = summary[summary["method"] == method]
        fig.add_scatter(
            x=sub["evaluation"], y=sub["mean_best_q45"] + sub["ci95"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
        fig.add_scatter(
            x=sub["evaluation"], y=sub["mean_best_q45"] - sub["ci95"],
            mode="lines", fill="tonexty", fillcolor=ci_fill[method],
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    fig.add_hline(y=80, line_dash="dash", line_color="#2a5f61", line_width=1.5,
                  annotation_text="Target 80%", annotation_font_color="#2a5f61")
    fig.add_hline(y=85, line_dash="dot", line_color="#c9a84c", line_width=1.5,
                  annotation_text="Stretch 85%", annotation_font_color="#c9a84c")
    fig.update_layout(**_PREMIUM_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    pivot = summary.pivot(index="evaluation", columns="method", values="mean_best_q45").reset_index()
    st.dataframe(pivot.round(3), use_container_width=True)


def main() -> None:
    initialize_state()
    apply_gnnbind_theme()
    df = st.session_state.experiments

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">⬡</div>
            <div>
                <div class="sidebar-brand-name">Formulation</div>
                <div class="sidebar-brand-sub">BO Explorer</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Design Space Explorer",
            "Run BO",
            "Experiment Log",
            "Convergence",
            "Compare",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small>API fixed at **30% w/w**<br>Excipients sum to **70% w/w**</small>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<h1>Formulation Bayesian Optimization <em>Explorer</em></h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Scientist-focused interface for model-based DoE — simulate dissolution outcomes, "
        "suggest next experiments, and compare optimization strategies."
    )
    render_header(df)

    if page == "Design Space Explorer":
        design_space_explorer(df)
    elif page == "Run BO":
        run_bo_page(df)
    elif page == "Experiment Log":
        experiment_log_page(df)
    elif page == "Convergence":
        convergence_page(df)
    elif page == "Compare":
        compare_page()


if __name__ == "__main__":
    main()
