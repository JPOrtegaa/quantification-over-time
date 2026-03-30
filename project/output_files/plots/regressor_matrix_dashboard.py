"""
Streamlit dashboard: TOMS regressor matrix M(t) over windows from CSV exports.

Tabs: (1) matrix M with regressor selection and optional classifier scores;
(2) true prevalence vs quantifier.

Run from the project folder:
  streamlit run output_files/plots/regressor_matrix_dashboard.py
"""

from __future__ import annotations

import colorsys
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as e:
    raise SystemExit(
        "Install plotly: pip install plotly streamlit"
    ) from e

try:
    import streamlit as st
except ImportError as e:
    raise SystemExit("Install streamlit: pip install streamlit") from e


PLOTS_DIR = Path(__file__).resolve().parent
OUTPUT_FILES = PLOTS_DIR.parent
REGRESSOR_DIR = OUTPUT_FILES / "regressor"
CLASSIFICATION_DIR = OUTPUT_FILES / "classification"
QUANTIFICATION_DIR = OUTPUT_FILES / "quantification"

# Cores base por índice na ordenação de rótulos; paletas específicas para [0,1] e [-1,0,1].
# Classificador: tom mais escuro na mesma família; M_r{i}_c{j}: tons conforme o regressor i.
CLASS_BASE_HEX_DEFAULT = (
    "#e67e22",
    "#27ae60",
    "#8e44ad",
    "#2980b9",
    "#c0392b",
)


def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.strip().lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(
        max(0, min(255, int(r * 255))),
        max(0, min(255, int(g * 255))),
        max(0, min(255, int(b * 255))),
    )


def _label_tuple_signature(class_labels: List) -> Tuple:
    return tuple(_sorted_labels(class_labels))


def palette_for_class_labels(class_labels: List) -> Tuple[str, ...]:
    """Mesma ordem que _sorted_labels / semantic_color_index."""
    sig = _label_tuple_signature(class_labels)
    try:
        sig_int = tuple(int(x) for x in sig)
    except (TypeError, ValueError):
        sig_int = None
    if sig_int == (0, 1):
        return ("#27ae60", "#e67e22")
    if sig_int == (-1, 0, 1):
        return ("#e67e22", "#27ae60", "#8e44ad")
    n = len(sig)
    if n <= len(CLASS_BASE_HEX_DEFAULT):
        return CLASS_BASE_HEX_DEFAULT[:n]
    return tuple(
        CLASS_BASE_HEX_DEFAULT[i % len(CLASS_BASE_HEX_DEFAULT)] for i in range(n)
    )


def _sorted_labels(class_labels: List) -> List:
    """Ordem estável para associar cada rótulo a um índice de cor."""

    def _key(x):
        if isinstance(x, bool):
            return (0, int(x))
        if isinstance(x, int):
            return (1, x)
        if isinstance(x, float):
            return (2, x)
        return (3, str(x))

    try:
        return sorted(class_labels, key=_key)
    except Exception:
        return sorted(class_labels, key=str)


def semantic_color_index(lab, class_labels: List) -> int:
    order = _sorted_labels(class_labels)
    try:
        return order.index(lab)
    except ValueError:
        return 0


def base_hex_for_semantic_label(lab, class_labels: List) -> str:
    pal = palette_for_class_labels(class_labels)
    k = semantic_color_index(lab, class_labels)
    return pal[k % len(pal)]


def matrix_row_shade(base_hex: str, row_i: int, n_rows: int) -> str:
    """Tons da mesma matiz: clareia/escurece conforme o regressor (linha i)."""
    r, g, b = _hex_to_rgb01(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if n_rows <= 1:
        l_new = max(0.22, min(0.72, l))
    else:
        t = row_i / (n_rows - 1)
        # i=0 mais escuro → i maior mais claro (ou o inverso); mantém contraste entre linhas
        l_new = 0.28 + t * 0.42
    s = min(0.95, max(0.45, s))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l_new, s)
    return _rgb01_to_hex(r2, g2, b2)


def classifier_shade_for_class_j(base_hex: str) -> str:
    """Traço do classificador: mesma família, traço bem visível (um pouco mais escuro/saturado)."""
    r, g, b = _hex_to_rgb01(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = max(0.2, min(0.55, l * 0.82))
    s = min(1.0, s * 1.08)
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return _rgb01_to_hex(r2, g2, b2)


def blend_hex(a: str, b: str, t: float) -> str:
    """t=0 → a, t=1 → b."""
    ar, ag, ab = _hex_to_rgb01(a)
    br, bg, bb = _hex_to_rgb01(b)
    u = 1.0 - t
    return _rgb01_to_hex(
        ar * u + br * t, ag * u + bg * t, ab * u + bb * t
    )


def matrix_j_for_score_column(col: str, class_labels: List) -> Optional[int]:
    short = col.replace("score_", "", 1)
    for j, lab in enumerate(class_labels):
        if str(lab) == short:
            return j
    return None


def matrix_j_for_prev_suffix(suf: str, class_labels: List) -> Optional[int]:
    for j, lab in enumerate(class_labels):
        if str(lab) == suf:
            return j
    return None


def _classifier_score_columns(df: pd.DataFrame) -> List[str]:
    """
    Classifier probability columns per class: ``score_<class>``.
    Excludes ``score_mean_*`` (row-mean of M in the regressor CSV).
    """
    out: List[str] = []
    for c in df.columns:
        s = str(c)
        if not s.startswith("score_"):
            continue
        if s.startswith("score_mean"):
            continue
        out.append(c)
    return sorted(out, key=str)


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


def classifier_short_label(meta: Dict[str, str]) -> str:
    slug = (meta.get("classifier_slug") or "").strip()
    return slug if slug else "clf"


def infer_class_labels(K: int) -> List:
    """
    Class label per regressor index (row i of M); aligned with energy loader [-1,0,1].
    If K≠3, uses 0..K-1.
    """
    if K == 3:
        return [-1, 0, 1]
    if K == 2:
        return [0, 1]
    return list(range(K))


def regressor_row_label(i: int, class_for_row) -> str:
    return f"Regressor {i} (trained only on y = {class_for_row})"


def discover_matrix_shape(df: pd.DataFrame) -> Tuple[int, int, List[Tuple[int, int]]]:
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


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def find_regressor_root_csvs(pattern: str = "regressor_window_scores_*.csv") -> List[Path]:
    if not REGRESSOR_DIR.is_dir():
        return []
    return sorted(REGRESSOR_DIR.glob(pattern))


def classifier_csv_for_regressor(reg_path: Path) -> Path:
    name = reg_path.name
    if not name.startswith("regressor_window_scores_"):
        return CLASSIFICATION_DIR / name
    return CLASSIFICATION_DIR / ("classifier_window_scores_" + name[len("regressor_window_scores_") :])


def quant_prev_csv_path(meta: Dict[str, str]) -> Path:
    ds = meta["dataset"]
    q = meta["quantifier"]
    seed = meta["seed"]
    clf = meta["classifier_slug"]
    return QUANTIFICATION_DIR / f"quant_prev_{ds}_{q}_seed{seed}_{clf}.csv"


def find_true_prevalence_csvs(dataset: str) -> List[Path]:
    if not QUANTIFICATION_DIR.is_dir():
        return []
    return sorted(QUANTIFICATION_DIR.glob(f"true_window_prevalence_{dataset}_v*.csv"))


def load_classifier_scores_aligned(
    reg_df: pd.DataFrame, reg_path: Path
) -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    if "window_index" not in reg_df.columns:
        return None, classifier_csv_for_regressor(reg_path)
    clf_path = classifier_csv_for_regressor(reg_path)
    if not clf_path.is_file():
        return None, clf_path
    try:
        clf = pd.read_csv(clf_path)
    except Exception:
        return None, clf_path
    if "window_index" not in clf.columns:
        return None, clf_path
    score_cols = _classifier_score_columns(clf)
    if not score_cols:
        return None, clf_path
    merged = reg_df.merge(
        clf[["window_index"] + score_cols],
        on="window_index",
        how="left",
    )
    return merged, clf_path


def tab_matrix_and_classifier(
    df: pd.DataFrame,
    x: pd.Series,
    x_label: str,
    K: int,
    class_labels: List,
    clf_aligned: Optional[pd.DataFrame],
    clf_path: Optional[Path],
    clf_lbl: str,
) -> None:
    st.markdown("### Matrix M(t) traces")
    st.caption(
        "Each **regressor** (row i) is trained only on examples whose true label is the one shown. "
        "When you select a regressor, all entries M_r{i}c0…M_r{i}c{K-1} are shown "
        "(regressor i’s prediction for each class j)."
    )

    row_options = list(range(K))
    row_labels = {
        i: regressor_row_label(i, class_labels[i]) for i in range(K)
    }
    selected_rows = st.multiselect(
        "Regressors to plot (M rows):",
        options=row_options,
        default=row_options,
        format_func=lambda i: row_labels[i],
    )

    show_clf = st.checkbox(
        f"Show classifier scores ({clf_lbl})",
        value=False,
        disabled=clf_aligned is None,
        help="Dashed traces: P(y=k) from the classification CSV.",
    )
    if clf_aligned is None and show_clf:
        show_clf = False

    fig = go.Figure()
    for i in selected_rows:
        for j in range(K):
            col = f"M_r{i}_c{j}"
            if col not in df.columns:
                continue
            lab_j = class_labels[j]
            base_lab = base_hex_for_semantic_label(lab_j, class_labels)
            line_color = matrix_row_shade(base_lab, i, K)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col],
                    mode="lines",
                    name=f"M r{i}→c{j} (y_train={class_labels[i]})",
                    legendgroup=f"r{i}",
                    line=dict(color=line_color, width=1.9),
                    showlegend=True,
                )
            )

    if show_clf and clf_aligned is not None:
        score_cols_plot = _classifier_score_columns(clf_aligned)
        for idx, col in enumerate(score_cols_plot):
            yc = clf_aligned[col]
            if yc.isna().all():
                continue
            short = col.replace("score_", "", 1)
            j = matrix_j_for_score_column(col, class_labels)
            if j is None:
                j = idx % max(1, K)
            lab_j = class_labels[j]
            base_lab = base_hex_for_semantic_label(lab_j, class_labels)
            color = classifier_shade_for_class_j(base_lab)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=yc,
                    mode="lines",
                    name=f"{clf_lbl} P(y={short})",
                    legendgroup="clf_scores",
                    line=dict(color=color, width=2.4, dash="dash"),
                    showlegend=True,
                )
            )

    n_m_traces = len(selected_rows) * K if selected_rows else 0
    fig.update_layout(
        title="M(t) and optional classifier scores",
        xaxis_title=x_label,
        yaxis_title="probability",
        height=min(720, 200 + 35 * max(1, n_m_traces + (3 if show_clf else 0))),
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    if len(fig.data) == 0:
        st.warning(
            "Select at least one regressor **or** enable classifier scores."
        )
    else:
        st.plotly_chart(fig, use_container_width=True)

    if clf_aligned is None and clf_path is not None:
        st.caption(
            f"Classifier: file at `{clf_path}` not found or missing `score_*` columns."
        )


def tab_prevalence(meta: Dict[str, str], class_labels: List) -> None:
    st.markdown("### True prevalence vs quantifier")
    dataset = meta.get("dataset", "")
    candidates = find_true_prevalence_csvs(dataset)
    if not candidates:
        st.warning(
            f"No `true_window_prevalence_{dataset}_v*.csv` in `{QUANTIFICATION_DIR}`. "
            "Run the experiment to generate `output_files/quantification/`."
        )
        return

    if len(candidates) > 1:
        true_path = st.selectbox(
            "True prevalence file:",
            options=candidates,
            format_func=lambda p: p.name,
        )
    else:
        true_path = candidates[0]

    quant_path = quant_prev_csv_path(meta)
    if not quant_path.is_file():
        st.error(
            f"Quantifier CSV not found: `{quant_path.name}`. "
            "The name must match the regressor run (quantifier, seed, classifier)."
        )
        try:
            st.caption(f"Full path: `{quant_path}`")
        except Exception:
            pass
        return

    try:
        true_df = pd.read_csv(true_path)
        quant_df = pd.read_csv(quant_path)
    except Exception as e:
        st.error(f"Error reading CSVs: {e}")
        return

    if "t_window" not in true_df.columns:
        st.error("Missing column `t_window` in the true-prevalence CSV.")
        return

    true_prev_cols = [c for c in true_df.columns if str(c).startswith("true_prev_")]
    quant_prev_cols = [c for c in quant_df.columns if str(c).startswith("quant_prev_")]
    if not true_prev_cols or not quant_prev_cols:
        st.error("Missing `true_prev_*` or `quant_prev_*` columns.")
        return

    merged = true_df.merge(
        quant_df[["window_index"] + quant_prev_cols],
        on="window_index",
        how="left",
    )
    x = merged["t_window"]

    fig = go.Figure()
    for idx, tcp in enumerate(sorted(true_prev_cols, key=str)):
        # true_prev_-1 -> class key "-1"
        suf = tcp.replace("true_prev_", "", 1)
        qcol = f"quant_prev_{suf}"
        if qcol not in merged.columns:
            continue
        j = matrix_j_for_prev_suffix(suf, class_labels)
        if j is None:
            j = idx % max(1, len(class_labels))
        lab_j = class_labels[j]
        base_lab = base_hex_for_semantic_label(lab_j, class_labels)
        # Mesma família de cor: verdadeiro mais forte; quantificador um tom mais claro (traço tracejado).
        ctrue = base_lab
        cquant = blend_hex(base_lab, "#ffffff", 0.38)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=merged[tcp],
                mode="lines",
                name=f"True P(y={suf})",
                line=dict(color=ctrue, width=2),
                legendgroup=f"cls{suf}",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=merged[qcol],
                mode="lines",
                name=f"Quantifier P(y={suf})",
                line=dict(color=cquant, width=2, dash="dash"),
                legendgroup=f"cls{suf}",
            )
        )

    fig.update_layout(
        title="Prevalence by class: true (solid) vs quantifier (dashed)",
        xaxis_title="t_window",
        yaxis_title="prevalence",
        height=520,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"True: `{true_path.name}` · Quantifier: `{quant_path.name}` "
        "(only test windows have `quant_prev_*`; validation has no dashed line)."
    )


def page_main():
    st.set_page_config(
        page_title="Regressor matrix M(t)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("TOMS regressor and quantification")
    st.caption(
        "Regressor: `output_files/regressor/`. Classification / quantification: "
        "`classification/` and `quantification/` folders."
    )

    files = find_regressor_root_csvs()
    if not files:
        st.error(
            f"No CSV files matching `regressor_window_scores_*.csv` in `{REGRESSOR_DIR}`."
        )
        return

    labels = [f.name for f in files]
    choice_idx = st.selectbox(
        "Regressor CSV file:",
        options=range(len(files)),
        format_func=lambda i: labels[i],
    )
    path = files[choice_idx]

    try:
        df = load_csv(path)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return

    if "t_window" not in df.columns:
        st.error(
            "Column `t_window` not found in this CSV. "
            "Add the column or use a compatible export."
        )
        return

    x = df["t_window"]
    x_label = "t_window"

    K, _, _ = discover_matrix_shape(df)
    if K == 0:
        st.error("No M_r*_c* columns found.")
        return

    meta = parse_regressor_csv_path(path)
    class_labels = infer_class_labels(K)
    st.subheader(f"`{path.name}` — K = {K}")
    with st.expander("File metadata"):
        st.json(meta)

    clf_aligned, clf_path = load_classifier_scores_aligned(df, path)
    clf_lbl = classifier_short_label(meta)
    if clf_aligned is not None:
        scols = _classifier_score_columns(clf_aligned)
        st.success(
            f"Classifier **{clf_lbl}** available (`{clf_path.name}`): {', '.join(scols)}."
        )
    else:
        st.info(
            f"No classification CSV at `{clf_path}` — only the prevalence tab may work "
            "if files exist under `quantification/`."
        )

    tab_m, tab_q = st.tabs(["Matrix M & classifier", "True prevalence vs quantifier"])

    with tab_m:
        tab_matrix_and_classifier(
            df, x, x_label, K, class_labels, clf_aligned, clf_path, clf_lbl
        )
        if "split" in df.columns:
            st.markdown("#### Rows by `split` (regressor)")
            st.bar_chart(df.groupby("split").size())

    with tab_q:
        tab_prevalence(meta, class_labels)


if __name__ == "__main__":
    page_main()
