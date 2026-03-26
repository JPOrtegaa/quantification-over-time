"""
TOMS: K regressores TimeSeriesMultinomial (um por classe), treinados só em linhas
cuja etiqueta verdadeira é essa classe; tempo por janela é escalar (um t por chunk).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd

from classification import Classifying
from regression import trainingModel as regression_trainingModel


def _time_column_to_days(time_series) -> np.ndarray:
    """Mesma convenção que quantifications._time_column_to_X (dias UNIX)."""
    s = time_series if isinstance(time_series, pd.Series) else pd.Series(time_series)
    t = pd.to_datetime(s, errors="coerce", dayfirst=True)
    if t.isna().any():
        mmdd = s.astype(str).str.strip() + "-2000"
        t_fallback = pd.to_datetime(mmdd, format="%m-%d-%Y", errors="coerce")
        t = t.where(~t.isna(), t_fallback)
    if t.isna().any():
        raise ValueError("Time column contains invalid or missing datetimes after parsing.")
    ns = t.astype("int64").to_numpy(dtype=np.float64)
    return (ns / (86400.0 * 1e9)).reshape(-1, 1)


@dataclass
class TOMSMultiRegressorBundle:
    """K regressores + mapa window_index -> t escalar (dias) + ordem das classes."""

    regressors: List[Any]
    window_t: Dict[int, float]
    classes: List


def scalar_time_per_window(ts_chunks, time_column: str) -> Dict[int, float]:
    """Um único t por janela: mediana dos tempos (em dias) de todas as linhas do chunk."""
    out = {}
    for wi in sorted(ts_chunks.keys()):
        df = ts_chunks[wi]
        if time_column not in df.columns:
            raise KeyError(
                f"time column {time_column!r} missing in window {wi}; "
                f"columns={list(df.columns)}"
            )
        raw_t = _time_column_to_days(df[time_column]).ravel()
        out[wi] = float(np.median(raw_t))
    return out


def attach_window_ids(val_set: pd.DataFrame, ts_chunks, val_length: int) -> pd.DataFrame:
    """Alinha linhas do val_set (concat de chunks 0..val_length-1) a índices de janela."""
    parts = [np.full(len(ts_chunks[w]), w, dtype=int) for w in range(val_length)]
    ids = np.concatenate(parts)
    if len(ids) != len(val_set):
        raise ValueError(
            f"val_set length {len(val_set)} != sum of val chunks {len(ids)}"
        )
    vs = val_set.copy().reset_index(drop=True)
    vs["_window_id"] = ids
    return vs


def _score_matrix_at_t(bundle: TOMSMultiRegressorBundle, t_scalar: float) -> np.ndarray:
    """Matriz (K, K): linha k = saída normalizada do regressor k (vetor sobre classes)."""
    K = len(bundle.regressors)
    M = np.zeros((K, K), dtype=np.float64)
    t_arr = np.array([float(t_scalar)], dtype=np.float64)
    for k, reg in enumerate(bundle.regressors):
        row = reg.predict(t_arr.reshape(-1, 1))
        row = np.asarray(row, dtype=np.float64).reshape(-1)
        if row.shape[0] != K:
            raise ValueError(f"Regressor {k} esperava {K} classes, obteve {row.shape}")
        row = np.clip(row, 1e-8, None)
        rs = row.sum()
        M[k] = row / rs if rs > 0 else 1.0 / K
    return M


def mean_simplex_from_matrix(M: np.ndarray) -> np.ndarray:
    """Média das K linhas de M, renormalizada no simplex (scores finais p/ quantificador)."""
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
    """CSV: uma linha por janela com t escalar, entradas M (KxK) e vetor média renormalizado."""
    rows = []
    K = len(classes)
    for wi in sorted(ts_chunks.keys()):
        t_w = bundle.window_t[wi]
        M = _score_matrix_at_t(bundle, t_w)
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


def fit_toms_multi_regressors(
    val_set: pd.DataFrame,
    classifier,
    classes: List,
    window_t: Dict[int, float],
    model_name: str,
    random_state: int,
    log_file: Optional[Path],
    **trainer_kw,
) -> TOMSMultiRegressorBundle:
    _, pred_scores, _ = Classifying.analyzer(val_set, classifier, classes)
    Y_all = pred_scores[[cl for cl in classes]].to_numpy(dtype=np.float64)
    true_y = val_set["label"].reset_index(drop=True).astype(int).to_numpy()
    window_ids = val_set["_window_id"].to_numpy()

    def _log(msg: str) -> None:
        if log_file is None:
            return
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as fp:
            fp.write(msg + "\n")

    K = len(classes)
    regressors: List[Any] = []

    _log("--- TRAIN: K regressors (one per class) ---")
    for k, cl in enumerate(classes):
        mask = true_y == cl
        n_k = int(mask.sum())
        if n_k == 0:
            _log(f"class={cl} (idx={k}): no samples; using constant uniform regressor.")

            class _UniformReg:
                def predict(self, X, _K=K):
                    X = np.asarray(X, dtype=np.float64)
                    n = X.shape[0] if X.ndim >= 2 else max(len(X.ravel()), 1)
                    return np.full((n, _K), 1.0 / _K, dtype=np.float64)

            regressors.append(_UniformReg())
            continue

        t_k = np.array([window_t[int(wi)] for wi in window_ids[mask]], dtype=np.float64)
        Y_k = Y_all[mask]
        _log(
            f"class={cl} (idx={k}): n={n_k}, t_min={t_k.min():.6g}, t_max={t_k.max():.6g}, "
            f"Y.shape={Y_k.shape}"
        )
        _log(f"  t_sample (first 5): {t_k[:5]}")
        _log(f"  Y_sample row0: {Y_k[0]}")

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
    Para cada linha: obtém M no t da janela; vetor de scores = média das linhas de M (simplex).
    Retorna (DataFrame de scores + true_y), e a lista de matrizes M por linha (para log opcional).
    """
    if "_window_id" not in df_text.columns:
        raise ValueError("scores_dataframe_from_bundle requer coluna '_window_id' no DataFrame.")
    wids = df_text["_window_id"].to_numpy()
    rows_scores = []
    matrices = []
    for i in range(len(df_text)):
        w = int(wids[i])
        t_w = bundle.window_t[w]
        M = _score_matrix_at_t(bundle, t_w)
        matrices.append(M)
        v = mean_simplex_from_matrix(M)
        rows_scores.append(v)
    mat = np.array(rows_scores)
    df_y = df_text["label"].reset_index(drop=True).astype(int)
    new_scores = pd.DataFrame({cl: mat[:, j] for j, cl in enumerate(classes)})
    new_scores["true_y"] = df_y.to_numpy()
    return new_scores, np.stack(matrices, axis=0) if matrices else np.empty((0, len(classes), len(classes)))


def analyzer_toms_multi_val(df_text, mod, classes, bundle: TOMSMultiRegressorBundle):
    """Validação TOMS: scores só dos K regressores (sem HF); rótulos preditos por argmax."""
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
    """Regista matrizes M por janela (val e test) e médias; no teste também média HF por janela."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    K = len(classes)

    with open(log_file, "a", encoding="utf-8") as fp:
        fp.write("--- VALIDATION: M matrix (KxK) per window (scalar t) ---\n")
        for w in range(val_length):
            t_w = bundle.window_t[w]
            M = _score_matrix_at_t(bundle, t_w)
            v = mean_simplex_from_matrix(M)
            fp.write(f"  window={w} split=val t={t_w:.8g}\n")
            fp.write(f"    M ({K}x{K}):\n{np.array2string(M, precision=6)}\n")
            fp.write(
                f"    mean_rows_renorm (scores fed to val quantifier): "
                f"{np.array2string(v, precision=6)}\n"
            )

        fp.write(
            "--- TEST: regressor M (log only); quantifier uses classifier scores ---\n"
        )
        for j in sorted(test_set_dict.keys()):
            df = test_set_dict[j]
            wi = j + val_length
            if wi not in bundle.window_t:
                raw_t = _time_column_to_days(df[time_column]).ravel()
                t_w = float(np.median(raw_t))
            else:
                t_w = bundle.window_t[wi]
            M = _score_matrix_at_t(bundle, t_w)
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
                f"    classifier mean in window (used by DyS on test): "
                f"{np.array2string(hf_mean, precision=6)}\n"
            )
