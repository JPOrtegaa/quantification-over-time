"""
TOMS: K TimeSeriesMultinomial regressors (one per class), each trained only on rows whose
true label is that class.

Training uses each row's parsed datetime as a continuous time (Unix epoch seconds, sub-day
resolution). Inference, CSV summaries, and quantifier calibration use one scalar per window:
the median of those seconds over all rows in the chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd

from methods.classification import Classifying
from methods.regression import trainingModel as regression_trainingModel


def _class_slug_for_filename(cl) -> str:
    """Safe fragment for paths (e.g. class -1 -> 'neg1' or 'm1')."""
    s = str(cl).strip()
    if s.startswith("-"):
        s = "neg" + s[1:]
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_") or "class"


def _per_regressor_train_csv_path(
    train_dir: Path, name_prefix: str, k: int, cl
) -> Path:
    slug = _class_slug_for_filename(cl)
    return Path(train_dir) / f"{name_prefix}_train_regressor_k{k}_y{slug}.csv"


def _build_per_regressor_train_rows(
    t_k: np.ndarray,
    Y_k: np.ndarray,
    classes: List,
) -> pd.DataFrame:
    """
    Columns that match regressor training input only: ``t_epoch_seconds`` + one column per
    class label (``str(c)``) with softmax scores from the classifier (shape Y_k (n, K)).
    """
    data: Dict[str, Any] = {"t_epoch_seconds": t_k.astype(np.float64)}
    for j, c in enumerate(classes):
        data[str(c)] = Y_k[:, j].astype(np.float64)
    return pd.DataFrame(data)


def _parse_time_series(time_series) -> pd.Series:
    """Parse time column to timezone-naive datetimes (same rules as legacy covid MM-DD)."""
    s = time_series if isinstance(time_series, pd.Series) else pd.Series(time_series)
    t = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if t.isna().any():
        mmdd = s.astype(str).str.strip() + "-2000"
        t_fallback = pd.to_datetime(mmdd, format="%m-%d-%Y", errors="coerce")
        t = t.where(~t.isna(), t_fallback)
    if t.isna().any():
        raise ValueError("Time column contains invalid or missing datetimes after parsing.")
    return t


def _time_column_to_epoch_seconds(time_series) -> np.ndarray:
    """
    Continuous time: nanoseconds since Unix epoch as float64 seconds (preserves hour/minute).
    Shape (n, 1).
    """
    t = _parse_time_series(time_series)
    ns = t.astype("int64").to_numpy(dtype=np.float64)
    return (ns / 1e9).reshape(-1, 1)


@dataclass
class TOMSMultiRegressorBundle:
    """K regressors + map window_index -> scalar t (median epoch seconds in window) + class order."""

    regressors: List[Any]
    window_t: Dict[int, float]
    classes: List


def scalar_time_per_window(ts_chunks, time_column: str) -> Dict[int, float]:
    """One scalar t per window: median of epoch seconds over all rows in the chunk."""
    out = {}
    for wi in sorted(ts_chunks.keys()):
        df = ts_chunks[wi]
        if time_column not in df.columns:
            raise KeyError(
                f"time column {time_column!r} missing in window {wi}; "
                f"columns={list(df.columns)}"
            )
        raw_t = _time_column_to_epoch_seconds(df[time_column]).ravel()
        out[wi] = float(np.median(raw_t))
    return out


def attach_window_ids(val_set: pd.DataFrame, ts_chunks, val_length: int) -> pd.DataFrame:
    """Map rows of val_set (concat of chunks 0..val_length-1) to window indices."""
    parts = [np.full(len(ts_chunks[w]), w, dtype=int) for w in range(val_length)]
    ids = np.concatenate(parts)
    if len(ids) != len(val_set):
        raise ValueError(
            f"val_set length {len(val_set)} != sum of val chunks {len(ids)}"
        )
    vs = val_set.copy().reset_index(drop=True)
    vs["_window_id"] = ids
    return vs


def score_matrix_at_t(bundle: TOMSMultiRegressorBundle, t_scalar: float) -> np.ndarray:
    """Matrix (K, K): row k is normalized output of regressor k (K-dim vector over classes)."""
    K = len(bundle.regressors)
    M = np.zeros((K, K), dtype=np.float64)
    t_arr = np.array([float(t_scalar)], dtype=np.float64)
    for k, reg in enumerate(bundle.regressors):
        row = reg.predict(t_arr.reshape(-1, 1))
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        if row.shape[0] != K:
            raise ValueError(f"Regressor {k} expected {K} classes, got {row.shape}")
        row = np.clip(row, 1e-8, None)
        rs = row.sum()
        M[k] = row / rs if rs > 0 else 1.0 / K
    return M


def mean_simplex_from_matrix(M: np.ndarray) -> np.ndarray:
    """Mean of the K rows of M, renormalized on the simplex (legacy summary scores)."""
    v = M.mean(axis=0)
    v = np.clip(v, 1e-8, None)
    s = v.sum()
    return v / s if s > 0 else np.full_like(v, 1.0 / len(v))


def write_bundle_window_scores_csv(
    ts_chunks,
    classes: List,
    bundle: TOMSMultiRegressorBundle,
    val_length: int,
    out_path: Path,
) -> None:
    """CSV: one row per window with scalar t, M (KxK) entries, and row-mean renormalized vector."""
    rows = []
    K = len(classes)
    for wi in sorted(ts_chunks.keys()):
        t_w = bundle.window_t[wi]
        M = score_matrix_at_t(bundle, t_w)
        v = mean_simplex_from_matrix(M)
        split = "val" if wi < val_length else "test"
        row = {
            "window_index": wi,
            "split": split,
            "n_samples": int(len(ts_chunks[wi])),
            "t_window": t_w,
        }
        for i in range(K):
            for j in range(K):
                row[f"M_r{i}_c{j}"] = float(M[i, j])
        for j, cl in enumerate(classes):
            row[f"score_mean_{cl}"] = float(v[j])
        rows.append(row)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_toms_test_window_csvs(
    bundle: TOMSMultiRegressorBundle,
    ts_chunks,
    classes: List,
    val_length: int,
    out_dir: Path,
    name_prefix: str,
) -> None:
    """
    One CSV per test window: scalar `t` (median epoch seconds) passed to the regressors
    and entries of M(t) (same convention as the bundle table).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    K = len(classes)
    for wi in sorted(ts_chunks.keys()):
        if wi < val_length:
            continue
        t_w = bundle.window_t[wi]
        M = score_matrix_at_t(bundle, t_w)
        row: Dict[str, Any] = {"t": float(t_w)}
        for i in range(K):
            for j in range(K):
                row[f"M_r{i}_c{j}"] = float(M[i, j])
        pd.DataFrame([row]).to_csv(
            out_dir / f"{name_prefix}_window_{wi}.csv", index=False
        )


def fit_toms_multi_regressors(
    val_set: pd.DataFrame,
    classifier,
    classes: List,
    window_t: Dict[int, float],
    time_column: str,
    model_name: str,
    random_state: int,
    log_file: Optional[Path],
    train_dir: Optional[Path] = None,
    train_name_prefix: str = "toms_train",
    **trainer_kw,
) -> TOMSMultiRegressorBundle:
    if time_column not in val_set.columns:
        raise KeyError(
            f"time column {time_column!r} missing in val_set; "
            f"columns={list(val_set.columns)}"
        )
    _, pred_scores, _ = Classifying.analyzer(val_set, classifier, classes)
    Y_all = pred_scores[[cl for cl in classes]].to_numpy(dtype=np.float64)
    true_y = val_set["label"].reset_index(drop=True).astype(int).to_numpy()
    t_all = _time_column_to_epoch_seconds(val_set[time_column]).ravel()

    def _log(msg: str) -> None:
        if log_file is None:
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as fp:
            fp.write(msg + "\n")

    K = len(classes)
    regressors: List[Any] = []
    train_dir_p = Path(train_dir) if train_dir is not None else None
    if train_dir_p is not None:
        train_dir_p.mkdir(parents=True, exist_ok=True)

    _log("--- TRAIN: K regressors (one per class) ---")
    for k, cl in enumerate(classes):
        mask = true_y == cl
        n_k = int(mask.sum())
        if train_dir_p is not None:
            csv_path = _per_regressor_train_csv_path(
                train_dir_p, train_name_prefix, k, cl
            )
        if n_k == 0:
            _log(f"class={cl} (idx={k}): no samples; using constant uniform regressor.")
            if train_dir_p is not None:
                empty_cols = ["t_epoch_seconds"] + [str(c) for c in classes]
                pd.DataFrame(columns=empty_cols).to_csv(csv_path, index=False)

            class _UniformReg:
                def predict(self, X, _K=K):
                    X = np.asarray(X, dtype=np.float64)
                    n = X.shape[0] if X.ndim >= 2 else max(len(X.ravel()), 1)
                    return np.full((n, _K), 1.0 / _K, dtype=np.float64)

            regressors.append(_UniformReg())
            continue

        t_k = t_all[mask].astype(np.float64)
        Y_k = Y_all[mask]
        _log(
            f"class={cl} (idx={k}): n={n_k}, t_min={t_k.min():.6g}, t_max={t_k.max():.6g}, "
            f"Y.shape={Y_k.shape}"
        )
        _log(f"  t_sample (first 5): {t_k[:5]}")
        _log(f"  Y_sample row0: {Y_k[0]}")

        if train_dir_p is not None:
            tr_df = _build_per_regressor_train_rows(t_k, Y_k, classes)
            tr_df.to_csv(csv_path, index=False)

        reg = regression_trainingModel.trainer(
            t_k.reshape(-1, 1),
            Y_k,
            model_name,
            random_state,
            **trainer_kw,
        )
        regressors.append(reg)

    bundle = TOMSMultiRegressorBundle(
        regressors=regressors, window_t=dict(window_t), classes=list(classes)
    )
    _log("--- End of train ---\n")
    return bundle


def scores_dataframe_from_bundle(
    df_text: pd.DataFrame, classes: List, bundle: TOMSMultiRegressorBundle
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    For each row: M at the row's window time; score vector = row-mean of M (simplex).
    Returns (score DataFrame + true_y), and stacked M matrices per row (optional logging).
    """
    if "_window_id" not in df_text.columns:
        raise ValueError(
            "scores_dataframe_from_bundle requires a '_window_id' column on the DataFrame."
        )
    wids = df_text["_window_id"].to_numpy()
    rows_scores = []
    matrices = []
    for i in range(len(df_text)):
        w = int(wids[i])
        t_w = bundle.window_t[w]
        M = score_matrix_at_t(bundle, t_w)
        matrices.append(M)
        v = mean_simplex_from_matrix(M)
        rows_scores.append(v)
    mat = np.array(rows_scores)
    df_y = df_text["label"].reset_index(drop=True).astype(int)
    new_scores = pd.DataFrame({cl: mat[:, j] for j, cl in enumerate(classes)})
    new_scores["true_y"] = df_y.to_numpy()
    return new_scores, np.stack(matrices, axis=0) if matrices else np.empty((0, len(classes), len(classes)))


def analyzer_toms_multi_val(df_text, mod, classes, bundle: TOMSMultiRegressorBundle):
    """TOMS validation scores from K regressors only (no HF); predicted labels via argmax."""
    new_scores, _ = scores_dataframe_from_bundle(df_text, classes, bundle)
    ty = new_scores["true_y"].to_numpy()
    adjusted = new_scores[[cl for cl in classes]].to_numpy()
    class_cols = np.array(classes)
    idx = np.argmax(adjusted, axis=1)
    new_pred_y = class_cols[idx]
    new_labels = pd.DataFrame({"pred_y": new_pred_y, "true_y": ty})
    return new_labels, new_scores, [0.0, 0.0]


def make_analyze_fn_toms_multi(bundle: TOMSMultiRegressorBundle):
    def analyze_fn(df_text, mod, classes):
        return analyzer_toms_multi_val(df_text, mod, classes, bundle)

    return analyze_fn


def log_validation_and_test_matrices(
    bundle: TOMSMultiRegressorBundle,
    val_set: pd.DataFrame,
    ts_chunks,
    test_set_dict: Dict,
    classifier,
    classes: List,
    time_column: str,
    val_length: int,
    log_file: Path,
) -> None:
    """Append per-window M matrices (val and test) and row-means; on test, also HF window mean."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    K = len(classes)

    with open(log_file, "a", encoding="utf-8") as fp:
        fp.write("--- VALIDATION: M matrix (KxK) per window (scalar t) ---\n")
        for w in range(val_length):
            t_w = bundle.window_t[w]
            M = score_matrix_at_t(bundle, t_w)
            v = mean_simplex_from_matrix(M)
            fp.write(f"  window={w} split=val t={t_w:.8g}\n")
            fp.write(f"    M ({K}x{K}):\n{np.array2string(M, precision=6)}\n")
            fp.write(
                f"    mean_rows_renorm (diagnostic only; val MAE uses M rows per val window): "
                f"{np.array2string(v, precision=6)}\n"
            )

        fp.write(
            "--- TEST: M from regressors used as quantifier calibration (K rows); "
            "document scores from classifier ---\n"
        )
        for j in sorted(test_set_dict.keys()):
            df = test_set_dict[j]
            wi = j + val_length
            if wi not in bundle.window_t:
                raw_t = _time_column_to_epoch_seconds(df[time_column]).ravel()
                t_w = float(np.median(raw_t))
            else:
                t_w = bundle.window_t[wi]
            M = score_matrix_at_t(bundle, t_w)
            v_reg = mean_simplex_from_matrix(M)
            _, pred_hf, _ = Classifying.analyzer(
                df, classifier, classes, hf_context=f"log w{wi}"
            )
            hf_mean = pred_hf[[cl for cl in classes]].mean(axis=0).to_numpy()
            hf_mean = np.clip(hf_mean, 1e-8, None)
            hf_mean = hf_mean / hf_mean.sum()
            fp.write(f"  window_index={wi} split=test t={t_w:.8g}\n")
            fp.write(f"    M regressor ({K}x{K}):\n{np.array2string(M, precision=6)}\n")
            fp.write(f"    mean_rows_renorm (log only): {np.array2string(v_reg, precision=6)}\n")
            fp.write(
                f"    classifier mean in window (quantifier test scores): "
                f"{np.array2string(hf_mean, precision=6)}\n"
            )
