"""
Streamlit dashboard: TOMS regressor matrix M(t) over windows from CSV exports.

Run from the project folder:
  streamlit run output_files/plots/regressor_matrix_dashboard.py

Optional:
  streamlit run output_files/plots/regressor_matrix_dashboard.py --server.headless true
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "Install plotly: pip install plotly streamlit"
    ) from e

try:
    import streamlit as st
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install streamlit: pip install streamlit") from e


PLOTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILES = PLOTS_DIR.parent
REGRESSOR_DIR = OUTPUT_FILES / "regressor"

KNOWN_QUANTIFIERS = (
    "DyS-Opt",
    "DyS",
    "ACC",
    "GPAC",
    "EDy",
    "CC",
    "ReadMe2",
)


def parse_regressor_csv_path(path: Path) -> Dict[str, str]:
    """
    Parse experiment metadata from filename:
    regressor_window_scores_{dataset}_{quantifier}_seed{n}_{classifier_slug}.csv
    """
    stem = path.stem
    out: Dict[str, str] = {"filename": path.name, "stem": stem}
    prefix = "regressor_window_scores_"
    if not stem.startswith(prefix):
        out["note"] = "Unrecognized prefix (expected regressor_window_scores_*)"
        return out

    rest = stem[len(prefix) :]
    m = re.match(r"(.+)_seed(\d+)_(.+)$", rest)
    if not m:
        out["note"] = "Could not parse seed_*_classifier tail"
        return out

    body, seed, clf_slug = m.group(1), m.group(2), m.group(3)
    dataset = ""
    quantifier = ""
    for q in sorted(KNOWN_QUANTIFIERS, key=len, reverse=True):
        suf = "_" + q
        if body.endswith(suf):
            dataset = body[: -len(suf)]
            quantifier = q
            break
    if not quantifier:
        idx = body.rfind("_")
        if idx >= 0:
            dataset, quantifier = body[:idx], body[idx + 1 :]
        else:
            dataset, quantifier = body, "?"

    out.update(
        {
            "dataset": dataset,
            "quantifier": quantifier,
            "seed": seed,
            "classifier_slug": clf_slug,
        }
    )
    return out


def discover_matrix_shape(df: pd.DataFrame) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Infer K from columns M_r{i}_c{j}."""
    pat = re.compile(r"^M_r(\d+)_c(\d+)$")
    cells: List[Tuple[int, int]] = []
    for c in df.columns:
        m = pat.match(str(c))
        if m:
            cells.append((int(m.group(1)), int(m.group(2))))
    if not cells:
        return 0, 0, []
    ki = max(i for i, _ in cells)
    kj = max(j for _, j in cells)
    K = max(ki, kj) + 1
    return K, K, cells


def build_matrix_row(df: pd.DataFrame, row_idx: int, K: int) -> np.ndarray:
    M = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            col = f"M_r{i}_c{j}"
            if col in df.columns:
                M[i, j] = float(df.iloc[row_idx][col])
    return M


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def page_main():
    st.set_page_config(
        page_title="Regressor matrix M(t)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("TOMS regressor matrix over time")
    st.caption(
        "Load `regressor_window_scores_*.csv` from `output_files/regressor/`. "
        "Each row is one time window; columns `M_r{i}_c{j}` are the K×K matrix for that window."
    )

    sidebar = st.sidebar
    csv_files = sorted(REGRESSOR_DIR.glob("regressor_window_scores_*.csv"))
    if not csv_files:
        sidebar.warning(f"No CSV files in `{REGRESSOR_DIR}`.")

    choice = sidebar.selectbox(
        "Regressor CSV",
        options=[str(p) for p in csv_files] if csv_files else [],
        format_func=lambda s: Path(s).name if s else "",
    )

    uploaded = sidebar.file_uploader("Or upload a CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        meta = {"filename": uploaded.name, "note": "Uploaded file"}
        path_label = uploaded.name
    elif choice:
        path = Path(choice)
        df = load_csv(path)
        meta = parse_regressor_csv_path(path)
        path_label = path.name
    else:
        st.info("Select or upload a regressor scores CSV to begin.")
        return

    with st.expander("Experiment metadata (from filename)", expanded=True):
        st.json(meta)

    K, _, _ = discover_matrix_shape(df)
    if K == 0:
        st.error("No M_r*_c* columns found.")
        return

    st.subheader(f"File: `{path_label}` — inferred K = {K}")

    x = df["window_index"] if "window_index" in df.columns else pd.Series(np.arange(len(df)))
    tcol = "t_window" if "t_window" in df.columns else None
    x_label = "window_index"

    tab_heat, tab_lines, tab_rows, tab_mean, tab_surface = st.tabs(
        [
            "Heatmap (window)",
            "All M entries vs window",
            "Rows over time",
            "Row-mean simplex",
            "Heatmap strip",
        ]
    )

    with tab_heat:
        win_i = st.slider(
            "Window row index (0-based in CSV)",
            0,
            max(0, len(df) - 1),
            min(7, len(df) - 1),
            key="win_slider",
        )
        M = build_matrix_row(df, win_i, K)
        split_lab = df.iloc[win_i].get("split", "") if "split" in df.columns else ""
        n_samp = df.iloc[win_i].get("n_samples", "")
        st.write(
            f"**Window** index `{df.iloc[win_i].get('window_index', win_i)}`, "
            f"split `{split_lab}`, n_samples `{n_samp}`"
            + (f", t_window `{df.iloc[win_i][tcol]}`" if tcol else "")
        )
        fig_h = go.Figure(
            data=go.Heatmap(
                z=M,
                x=[f"c{j}" for j in range(K)],
                y=[f"r{i}" for i in range(K)],
                colorscale="Viridis",
                zmin=0,
                zmax=1,
                text=np.round(M, 3),
                texttemplate="%{text}",
            )
        )
        fig_h.update_layout(
            title="M matrix (rows = regressors / classes, cols = class probabilities)",
            xaxis_title="column (class)",
            yaxis_title="row (regressor)",
            height=420,
        )
        st.plotly_chart(fig_h, use_container_width=True)

    with tab_lines:
        st.markdown("Time series of every matrix entry (can be busy for large K).")
        fig_l = go.Figure()
        for i in range(K):
            for j in range(K):
                col = f"M_r{i}_c{j}"
                if col in df.columns:
                    fig_l.add_trace(
                        go.Scatter(
                            x=x.values if hasattr(x, "values") else x,
                            y=df[col],
                            mode="lines",
                            name=f"r{i}c{j}",
                            legendgroup=f"g{i}{j}",
                            showlegend=(K <= 4),
                        )
                    )
        fig_l.update_layout(
            title="M_{i,j} vs window index",
            xaxis_title=x_label,
            yaxis_title="probability",
            height=min(540, 120 + 40 * K * K),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        if K > 4:
            st.caption("Legend hidden for K>4; use Plotly toolbar to isolate traces.")
            fig_l.update_layout(showlegend=False)
        st.plotly_chart(fig_l, use_container_width=True)

    with tab_rows:
        st.markdown("One subplot per **row** of M (each regressor’s softmax over classes).")
        fig_r = make_subplots(
            rows=K,
            cols=1,
            subplot_titles=[f"Row {i} (regressor for class axis {i})" for i in range(K)],
            vertical_spacing=0.06,
        )
        for i in range(K):
            for j in range(K):
                col = f"M_r{i}_c{j}"
                if col in df.columns:
                    fig_r.add_trace(
                        go.Scatter(
                            x=x.values if hasattr(x, "values") else x,
                            y=df[col],
                            name=f"col j={j}",
                            showlegend=(i == 0),
                        ),
                        row=i + 1,
                        col=1,
                    )
        fig_r.update_layout(height=200 * K, title_text="Rows of M vs window")
        fig_r.update_xaxes(title_text=x_label, row=K, col=1)
        st.plotly_chart(fig_r, use_container_width=True)

    with tab_mean:
        mean_cols = [c for c in df.columns if str(c).startswith("score_mean_")]
        if not mean_cols:
            st.warning("No score_mean_* columns in this CSV.")
        else:
            fig_m = go.Figure()
            xv = x.values if hasattr(x, "values") else x
            for c in mean_cols:
                fig_m.add_trace(go.Scatter(x=xv, y=df[c], mode="lines+markers", name=c))
            fig_m.update_layout(
                title="Row-mean renormalized vector (mean simplex mix) over windows",
                xaxis_title=x_label,
                yaxis_title="probability",
                height=440,
            )
            st.plotly_chart(fig_m, use_container_width=True)

    with tab_surface:
        st.markdown(
            "**3D surface:** one row of M across windows. **Strip heatmap:** all windows × "
            "flattened entries (validation vs test if `split` is in the CSV)."
        )
        row_pick = st.selectbox(
            "Surface: which matrix row (regressor index)",
            list(range(K)),
            index=0,
            key="surface_row",
        )
        Zr = np.array([build_matrix_row(df, r, K)[row_pick, :] for r in range(len(df))])
        n_win = len(df)
        xs = np.arange(K)
        ys = np.arange(n_win)
        fig_s2 = go.Figure(
            data=go.Surface(
                x=xs,
                y=ys,
                z=Zr,
                colorscale="Plasma",
            )
        )
        fig_s2.update_layout(
            title=f"Surface: M[row={row_pick}, :] — x=class column, y=window (CSV row)",
            scene=dict(
                xaxis_title="class column j",
                yaxis_title="window (CSV row index)",
                zaxis_title="M value",
            ),
            height=520,
        )
        st.plotly_chart(fig_s2, use_container_width=True)

        flat = np.array([build_matrix_row(df, r, K).reshape(-1) for r in range(len(df))])
        labels = [f"r{i}c{j}" for i in range(K) for j in range(K)]
        xv = x.values if hasattr(x, "values") else np.asarray(x)
        fig_strip = go.Figure(
            data=go.Heatmap(
                z=flat,
                x=labels,
                y=xv.astype(str),
                colorscale="Turbo",
                zmin=0,
                zmax=1,
            )
        )
        fig_strip.update_layout(
            title="Heatmap: window × flattened M entries",
            xaxis_title="entry",
            yaxis_title=x_label,
            height=max(320, min(900, 6 * len(df))),
        )
        st.plotly_chart(fig_strip, use_container_width=True)

    if "split" in df.columns:
        st.subheader("Splits")
        st.bar_chart(df.groupby("split").size())


if __name__ == "__main__":
    page_main()
