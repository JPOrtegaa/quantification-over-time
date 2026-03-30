"""
TOMS: K TimeSeriesMultinomial regressors (one per class), each trained only on rows whose
true label is that class.

Time features (see ``time_encoding``):

* ``scalar`` (default): each row uses Unix epoch seconds; each window uses the median of
  those seconds for inference.
* ``week``: 7 columns one-hot for weekday (pandas ``dayofweek``: Monday=0 … Sunday=6);
  training uses per-row one-hot; inference uses one-hot of the median calendar day in the chunk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

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
    X_k: np.ndarray,
    Y_k: np.ndarray,
    classes: List,
    time_encoding: str,
) -> pd.DataFrame:
    """
    Columns that match regressor training input + one column per class (softmax from clf).
    Scalar mode: ``t_epoch_seconds``. Week mode: ``dow_0``…``dow_6`` (Mon…Sun).
    """
    data: Dict[str, Any] = {}
    if time_encoding == "week":
        for d in range(7):
            data[f"dow_{d}"] = X_k[:, d].astype(np.float64)
    else:
        data["t_epoch_seconds"] = np.asarray(X_k, dtype=np.float64).ravel()
    for j, c in enumerate(classes):
        data[str(c)] = Y_k[:, j].astype(np.float64)
    return pd.DataFrame(data)


def _parse_time_series(time_series) -> pd.Series:
    """Parse time column to timezone-naive datetimes (same rules as legacy covid MM-DD)."""
    s = time_series if isinstance(time_series, pd.Series) else pd.Series(time_series)
    strs = s.astype(str).str.strip()
    # ISO calendar dates (tabular energy, etc.): avoid dayfirst=True — it can mis-parse
    # YYYY-MM-DD in some pandas builds and break monotonic t_window medians.
    iso_mask = strs.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    if bool(iso_mask.all()):
        t = pd.to_datetime(strs, format="%Y-%m-%d", errors="coerce")
    else:
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


def _weekday_one_hot_from_series(time_series) -> np.ndarray:
    """One row per sample: Monday=0 … Sunday=6 → 7 columns. Shape (n, 7)."""
    t = _parse_time_series(time_series)
    dows = t.dt.dayofweek.to_numpy(dtype=int)
    n = len(dows)
    X = np.zeros((n, 7), dtype=np.float64)
    X[np.arange(n, dtype=int), dows] = 1.0
    return X


def _window_median_weekday_one_hot(
    ts_chunks, time_column: str
) -> Dict[int, np.ndarray]:
    """Per window: one-hot of weekday of median timestamp in chunk. Values shape (1, 7)."""
    out: Dict[int, np.ndarray] = {}
    for wi in _sorted_window_keys(ts_chunks):
        df = ts_chunks[wi]
        if time_column not in df.columns:
            raise KeyError(
                f"time column {time_column!r} missing in window {wi}; "
                f"columns={list(df.columns)}"
            )
        raw_t = _parse_time_series(df[time_column])
        med = raw_t.median()
        dow = int(pd.Timestamp(med).dayofweek)
        v = np.zeros((1, 7), dtype=np.float64)
        v[0, dow] = 1.0
        out[wi] = v
    return out


def build_window_row_features(
    ts_chunks,
    time_column: str,
    time_encoding: str,
    window_t: Dict[int, float],
) -> Dict[int, np.ndarray]:
    """
    Per window index, feature row passed to sklearn regressors at inference: (1, 1) epoch seconds
    or (1, 7) weekday one-hot.
    """
    if time_encoding == "week":
        return _window_median_weekday_one_hot(ts_chunks, time_column)
    return {wi: np.array([[window_t[wi]]], dtype=np.float64) for wi in window_t}


@dataclass
class TOMSMultiRegressorBundle:
    """K regressors + per-window time features + median epoch (for exports) + class order."""

    regressors: List[Any]
    window_t: Dict[int, float]
    classes: List
    window_row_features: Dict[int, np.ndarray]
    time_encoding: str = "scalar"


def _sorted_window_keys(ts_chunks) -> List:
    """Sort chunk keys numerically when possible (avoids '10' before '2' if keys were str)."""

    def _key(k: Any) -> Union[int, Any]:
        if isinstance(k, (int, np.integer)):
            return int(k)
        if isinstance(k, str) and k.isdigit():
            return int(k)
        return k

    return sorted(ts_chunks.keys(), key=_key)


def diagnose_scalar_time_monotonicity(
    ts_chunks, time_column: str
) -> Dict[str, Any]:
    """
    Check whether median t per window is non-decreasing in window key order.
    Call after scalar_time_per_window to validate exports / dashboards.
    """
    wt = scalar_time_per_window(ts_chunks, time_column)
    order = _sorted_window_keys(ts_chunks)
    violations: List[Tuple[Any, Any, float, float]] = []
    for a, b in zip(order, order[1:]):
        if wt[b] + 1e-9 < wt[a]:
            violations.append((a, b, float(wt[a]), float(wt[b])))
    return {
        "time_column": time_column,
        "n_windows": len(order),
        "monotonic_non_decreasing": len(violations) == 0,
        "violations": violations[:50],
    }


def scalar_time_per_window(ts_chunks, time_column: str) -> Dict[int, float]:
    """One scalar t per window: median of epoch seconds over all rows in the chunk."""
    out = {}
    for wi in _sorted_window_keys(ts_chunks):
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


def score_matrix_at_window(bundle: TOMSMultiRegressorBundle, wi: int) -> np.ndarray:
    """Matrix (K, K): row k is normalized output of regressor k (K-dim vector over classes)."""
    K = len(bundle.regressors)
    M = np.zeros((K, K), dtype=np.float64)
    X = np.asarray(bundle.window_row_features[wi], dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    for k, reg in enumerate(bundle.regressors):
        row = reg.predict(X)
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
    for wi in _sorted_window_keys(ts_chunks):
        t_w = bundle.window_t[wi]
        M = score_matrix_at_window(bundle, wi)
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
    for wi in _sorted_window_keys(ts_chunks):
        if wi < val_length:
            continue
        t_w = bundle.window_t[wi]
        M = score_matrix_at_window(bundle, wi)
        row: Dict[str, Any] = {"t": float(t_w)}
        if bundle.time_encoding == "week":
            oh = np.asarray(bundle.window_row_features[wi], dtype=np.float64).ravel()
            for d in range(7):
                row[f"dow_{d}"] = float(oh[d])
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
    window_row_features: Optional[Dict[int, np.ndarray]] = None,
    time_encoding: str = "scalar",
    **trainer_kw,
) -> TOMSMultiRegressorBundle:
    if time_encoding not in ("scalar", "week"):
        raise ValueError(f"time_encoding must be 'scalar' or 'week', got {time_encoding!r}")
    if time_column not in val_set.columns:
        raise KeyError(
            f"time column {time_column!r} missing in val_set; "
            f"columns={list(val_set.columns)}"
        )
    if window_row_features is None:
        if time_encoding != "scalar":
            raise ValueError(
                "window_row_features must be provided when time_encoding is 'week'"
            )
        window_row_features = {
            wi: np.array([[window_t[wi]]], dtype=np.float64) for wi in window_t
        }
    wrf = {int(wi): np.asarray(v, dtype=np.float64) for wi, v in window_row_features.items()}

    _, pred_scores, _ = Classifying.analyzer(val_set, classifier, classes)
    Y_all = pred_scores[[cl for cl in classes]].to_numpy(dtype=np.float64)
    true_y = val_set["label"].reset_index(drop=True).astype(int).to_numpy()
    if time_encoding == "week":
        X_all = _weekday_one_hot_from_series(val_set[time_column])
    else:
        X_all = _time_column_to_epoch_seconds(val_set[time_column])

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
                if time_encoding == "week":
                    empty_cols = [f"dow_{d}" for d in range(7)] + [str(c) for c in classes]
                else:
                    empty_cols = ["t_epoch_seconds"] + [str(c) for c in classes]
                pd.DataFrame(columns=empty_cols).to_csv(csv_path, index=False)

            class _UniformReg:
                def predict(self, X, _K=K):
                    X = np.asarray(X, dtype=np.float64)
                    n = X.shape[0] if X.ndim >= 2 else max(len(X.ravel()), 1)
                    return np.full((n, _K), 1.0 / _K, dtype=np.float64)

            regressors.append(_UniformReg())
            continue

        X_k = X_all[mask].astype(np.float64)
        Y_k = Y_all[mask]
        _log(
            f"class={cl} (idx={k}): n={n_k}, time_encoding={time_encoding!r}, "
            f"X.shape={X_k.shape}, Y.shape={Y_k.shape}"
        )
        _log(f"  X_sample row0: {X_k[0]}")
        _log(f"  Y_sample row0: {Y_k[0]}")

        if train_dir_p is not None:
            tr_df = _build_per_regressor_train_rows(X_k, Y_k, classes, time_encoding)
            tr_df.to_csv(csv_path, index=False)

        reg = regression_trainingModel.trainer(
            X_k,
            Y_k,
            model_name,
            random_state,
            **trainer_kw,
        )
        regressors.append(reg)

    bundle = TOMSMultiRegressorBundle(
        regressors=regressors,
        window_t=dict(window_t),
        classes=list(classes),
        window_row_features=wrf,
        time_encoding=time_encoding,
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
        M = score_matrix_at_window(bundle, w)
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
        fp.write("--- VALIDATION: M matrix (KxK) per window ---\n")
        for w in range(val_length):
            t_w = bundle.window_t[w]
            M = score_matrix_at_window(bundle, w)
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
            t_w = bundle.window_t.get(wi)
            if t_w is None:
                raw_t = _time_column_to_epoch_seconds(df[time_column]).ravel()
                t_w = float(np.median(raw_t))
            M = score_matrix_at_window(bundle, wi)
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
