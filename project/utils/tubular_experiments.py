"""
Tabular time-series quantification (bike, energy, news): sklearn classifiers, optional
TOMS (same multi-regressor path as textual), and ``original`` mode as in
``run_original.experiment``. Uses ``data_loading_old``.

For energy, chunks keep the CSV ``date`` column (YYYY-MM-DD); set
``REGRESSOR_TIME_COLUMN = "date"`` in VARIABLES. That column is used for TOMS time
features and is excluded from sklearn inputs via ``DataFrame.attrs``. For TOMS, use
``REGRESSOR_TIME_ENCODING = "scalar"`` (epoch) or ``"week"`` (7-column weekday one-hot).
"""

from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
import data_loading_old as data_loading
import quantifications as qfy
import utils
from methods.classification import Classifying
from methods.classification import trainingModel
from methods.regression.toms_multi_regressor import (
    TOMSMultiRegressorBundle,
    attach_window_ids,
    build_window_row_features,
    diagnose_scalar_time_monotonicity,
    fit_toms_multi_regressors,
    log_validation_and_test_matrices,
    scalar_time_per_window,
    write_bundle_window_scores_csv,
    write_toms_test_window_csvs,
)
from time_series_adjustment import KalmanMA, MovingAverage
from utils import params_KFMA
from utils.main_functions_experiments import (
    TextualExperimentConfig,
    annotate_best_method,
    compute_initial_window_and_split,
    run_test_quantification,
    run_validation_quantification,
    tsa_adjust_compute_mae,
)

warnings.filterwarnings("ignore")


def _stamp_sklearn_exclude_cols(
    training_set: pd.DataFrame, ts_chunks: dict, *cols: str
) -> None:
    """Mark columns to omit from sklearn ``predict`` (e.g. calendar ``date`` for TOMS)."""
    merged = tuple(dict.fromkeys(c for c in cols if c))
    if not merged:
        return
    for df in (training_set, *ts_chunks.values()):
        try:
            df.attrs["sklearn_exclude_cols"] = merged
        except AttributeError:
            pass


def _training_feature_matrix(
    training_set: pd.DataFrame, time_column: str | None = None
) -> pd.DataFrame:
    """Drop label, window id, time column, and ``_toms_*`` (not sklearn features)."""
    drop = {"label"}
    if "_window_id" in training_set.columns:
        drop.add("_window_id")
    if time_column and time_column in training_set.columns:
        drop.add(time_column)
    for c in training_set.columns:
        if str(c).startswith("_toms_"):
            drop.add(c)
    return training_set.loc[:, ~training_set.columns.isin(drop)]


def tubular_single_experiment(
    dataset: Tuple[str, int],
    classifier_name: str,
    quantifier: str,
    tsa: str,
    random_state: int,
    unified_window: int,
    time_column: str,
) -> float:
    """Original mode: train sklearn clf, then val/test quantification + TSA."""
    training_set, ts_chunks, ts_prevalence, c, ts_info = data_loading.loading(dataset[0])
    _stamp_sklearn_exclude_cols(training_set, ts_chunks, time_column)
    classifier = trainingModel.trainer(
        _training_feature_matrix(training_set, time_column),
        training_set["label"],
        classifier_name,
        random_state,
    )

    if dataset[1] < unified_window:
        lf = 0
    else:
        lf = dataset[1] - unified_window
    inital_value = ts_prevalence.iloc[lf : dataset[1], :].to_numpy()
    val_true = ts_prevalence[: dataset[1]].to_numpy()
    val_set, test_sets, test_dsts = utils.val_test_split(
        ts_chunks.copy(), ts_prevalence, dataset[1]
    )
    try:
        val_set.attrs["sklearn_exclude_cols"] = (time_column,)
    except AttributeError:
        pass

    true_prev_path = (
        config.OUTPUT_QUANTIFICATION_DIR
        / f"true_window_prevalence_{dataset[0]}_v{dataset[1]}.csv"
    )
    qfy.write_true_window_prevalence_csv(
        ts_prevalence,
        ts_chunks,
        c,
        dataset[1],
        true_prev_path,
        time_column=time_column,
    )

    val_MAE, val_MSE, sep_mae, val_pred_dists = qfy.getMAE_val_set(
        val_set,
        quantifier,
        classifier,
        c,
        ts_chunks,
        dataset,
        ts_info,
        random_seed=random_state,
    )

    quantified_dsts = qfy.qtfied_dists(
        val_set,
        test_sets,
        dataset,
        quantifier,
        classifier,
        c,
        ts_info,
        random_seed=random_state,
    )
    clf_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(classifier_name))[:80]
    quant_prev_path = (
        config.OUTPUT_QUANTIFICATION_DIR
        / f"quant_prev_{dataset[0]}_{quantifier}_seed{random_state}_{clf_slug}.csv"
    )
    qfy.write_quantified_prevalence_csv(
        quantified_dsts,
        c,
        dataset[1],
        quant_prev_path,
        ts_chunks=ts_chunks,
        time_column=time_column,
    )
    qua_mae = utils.mae(test_dsts, quantified_dsts)

    if tsa == "QFY":
        return qua_mae

    modified_dsts = []
    val_init_value = np.empty((0, len(c)))
    validation = [val_init_value, val_pred_dists, val_true, c, unified_window]
    if tsa == "MA":
        for index in range(len(c)):
            modified_prevs = MovingAverage(
                initial_value=inital_value[:, index],
                quantified_prevs=quantified_dsts[:, index],
                window=unified_window,
            )
            modified_dsts.append(modified_prevs)
            if len(c) == 2:
                modified_dsts.append(1 - modified_prevs)
                break

    elif tsa == "KFMA":
        _, _q = params_KFMA(validation, val_MSE)
        q = 10**_q
        for index in range(len(c)):
            modified_prevs = KalmanMA(
                initial_value=inital_value[:, index],
                observations=quantified_dsts[:, index],
                qtfy_error=val_MSE,
                state_dim=unified_window,
                q=q,
            )
            modified_dsts.append(modified_prevs)
    else:
        raise ValueError(f"Unknown TSA {tsa!r}")

    modified_dsts = np.array(modified_dsts).T
    modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))
    return utils.mae(test_dsts, modified_dsts)


def tubular_toms_experiment(
    cfg: TextualExperimentConfig,
    dataset: Tuple[str, int],
    classifier_name: str,
    quantifier: str,
    tsa: str,
    random_state: int,
) -> float:
    """TOMS branch: sklearn clf + K time regressors (same contract as textual TOMS)."""
    if tsa != "QFY":
        raise ValueError("TOMS only supports tsa='QFY'")

    time_col = cfg.regressor_time_column
    training_set, ts_chunks, ts_prevalence, c, ts_info = data_loading.loading(dataset[0])
    ts_chunks = {k: v.copy() for k, v in ts_chunks.items()}
    _stamp_sklearn_exclude_cols(training_set, ts_chunks, time_col)

    sample = next(iter(ts_chunks.values()))
    if time_col not in sample.columns:
        raise ValueError(
            f"TOMS tabular requires time column {time_col!r} in every chunk; "
            f"got columns {list(sample.columns)}. For energy, use REGRESSOR_TIME_COLUMN = 'date'."
        )

    classifier = trainingModel.trainer(
        _training_feature_matrix(training_set, time_col),
        training_set["label"],
        classifier_name,
        random_state,
    )

    def _log(msg: str) -> None:
        print(f"{cfg.log_prefix} {msg}", flush=True)

    _log(
        "=== tubular TOMS === "
        f"dataset={dataset[0]!r}, quantifier={quantifier!r}, tsa={tsa!r}, "
        f"seed={random_state}, classifier={classifier_name!r}, time_col={time_col!r}"
    )

    inital_value, val_true, val_set, test_sets, test_dsts = (
        compute_initial_window_and_split(cfg, dataset, ts_chunks, ts_prevalence)
    )

    true_prev_path = (
        config.OUTPUT_QUANTIFICATION_DIR
        / f"true_window_prevalence_{dataset[0]}_v{dataset[1]}.csv"
    )
    qfy.write_true_window_prevalence_csv(
        ts_prevalence,
        ts_chunks,
        c,
        dataset[1],
        true_prev_path,
        time_column=time_col,
    )
    _log(f"True per-window prevalence → {true_prev_path}")

    window_t = scalar_time_per_window(ts_chunks, time_col)
    window_row_features = build_window_row_features(
        ts_chunks, time_col, cfg.regressor_time_encoding, window_t
    )
    mono = diagnose_scalar_time_monotonicity(ts_chunks, time_col)
    if mono["monotonic_non_decreasing"]:
        _log(f"t_window medians: non-decreasing in window order ({mono['n_windows']} windows).")
    else:
        _log(
            "WARNING: t_window medians are not non-decreasing in window key order — "
            f"first violations (up to 5): {mono['violations'][:5]}"
        )
    val_set = attach_window_ids(val_set, ts_chunks, dataset[1])
    try:
        val_set.attrs["sklearn_exclude_cols"] = (time_col,)
    except AttributeError:
        pass

    clf_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(classifier_name))[:80]
    clf_out = (
        config.OUTPUT_CLASSIFICATION_DIR
        / f"classifier_window_scores_{dataset[0]}_{quantifier}_seed{random_state}_{clf_slug}.csv"
    )
    Classifying.HF_PHASE_HINT = "clf CSV / window"
    try:
        qfy.write_classifier_window_scores_table(
            ts_chunks,
            c,
            classifier,
            val_length=dataset[1],
            out_path=clf_out,
            time_column=time_col,
        )
    finally:
        Classifying.HF_PHASE_HINT = None
    _log(f"Classifier window score table saved to {clf_out}.")

    regressor: Optional[TOMSMultiRegressorBundle] = None
    log_path = cfg.regressor_log_path
    ts_run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(
            f"\n{'=' * 72}\n"
            f"TOMS multi-regressor (tubular) | utc={ts_run}\n"
            f"dataset={dataset[0]!r} quantifier={quantifier!r} seed={random_state}\n"
            f"classifier={classifier_name!r}\n"
            f"window_t (first 5): "
            f"{ {w: window_t[w] for w in sorted(window_t.keys())[:5]} }\n"
        )

    reg_out = (
        config.OUTPUT_REGRESSOR_DIR
        / f"regressor_window_scores_{dataset[0]}_{quantifier}_seed{random_state}_{clf_slug}.csv"
    )
    train_reg_dir = config.OUTPUT_REGRESSOR_DIR / "train"
    test_reg_dir = config.OUTPUT_REGRESSOR_DIR / "test"
    Classifying.HF_PHASE_HINT = "TOMS train: val Y (tabular)"
    try:
        span = qfy.date_span_label(val_set, time_col)
        if span:
            try:
                val_set.attrs["hf_log"] = span
            except AttributeError:
                pass
        regressor = fit_toms_multi_regressors(
            val_set,
            classifier,
            c,
            window_t,
            time_col,
            cfg.regressor_label,
            random_state,
            log_path,
            train_dir=train_reg_dir,
            train_name_prefix=reg_out.stem,
            window_row_features=window_row_features,
            time_encoding=cfg.regressor_time_encoding,
            **cfg.regressor_tsmn_kwargs,
        )
    finally:
        Classifying.HF_PHASE_HINT = None
    _log("TOMS: regressors trained (tubular).")
    write_bundle_window_scores_csv(
        ts_chunks, c, regressor, val_length=dataset[1], out_path=reg_out
    )
    write_toms_test_window_csvs(
        regressor,
        ts_chunks,
        c,
        val_length=dataset[1],
        out_dir=test_reg_dir,
        name_prefix=reg_out.stem,
    )

    Classifying.HF_PHASE_HINT = f"val MAE | {quantifier}"
    try:
        val_MAE, val_MSE, sep_mae, val_pred_dists = run_validation_quantification(
            cfg,
            val_set,
            quantifier,
            classifier,
            c,
            ts_chunks,
            dataset,
            ts_info,
            random_state,
            regressor=regressor,
            time_column=time_col,
        )
    finally:
        Classifying.HF_PHASE_HINT = None

    Classifying.HF_PHASE_HINT = f"test qtfy | {quantifier}"
    try:
        quantified_dsts = run_test_quantification(
            cfg,
            val_set,
            test_sets,
            dataset,
            quantifier,
            classifier,
            c,
            ts_info,
            random_state,
            regressor=regressor,
            time_column=time_col,
        )
    finally:
        Classifying.HF_PHASE_HINT = None

    quant_prev_path = (
        config.OUTPUT_QUANTIFICATION_DIR
        / f"quant_prev_{dataset[0]}_{quantifier}_seed{random_state}_{clf_slug}.csv"
    )
    qfy.write_quantified_prevalence_csv(
        quantified_dsts,
        c,
        dataset[1],
        quant_prev_path,
        ts_chunks=ts_chunks,
        time_column=time_col,
    )
    _log(f"Quantifier prevalence (test windows) → {quant_prev_path}")

    if regressor is not None:
        log_validation_and_test_matrices(
            regressor,
            val_set,
            ts_chunks,
            test_sets,
            classifier,
            c,
            time_col,
            dataset[1],
            cfg.regressor_log_path,
        )

    mae_out = tsa_adjust_compute_mae(
        cfg,
        tsa,
        quantified_dsts,
        test_dsts,
        inital_value,
        val_pred_dists,
        val_true,
        c,
        val_MSE,
    )
    _log(f"=== end tubular TOMS (MAE) = {mae_out:.6f} ===")
    return mae_out


def run_tubular_experiments_grid(
    *,
    dataset_name: str,
    val_length: int,
    seeds: Tuple[int, ...],
    qua_methods: Tuple[str, ...],
    tsa_methods: Tuple[str, ...],
    classifiers: Tuple[str, ...],
    exp_types: Tuple[str, ...],
    cfg_toms: TextualExperimentConfig,
    time_column: str,
    unified_window: int,
    log_prefix: str,
    quick: bool,
) -> None:
    """Same row layout as textual grid (Dataset, ExpType, …); averages over seeds."""
    dataset = (dataset_name, val_length)

    def log(msg: str) -> None:
        print(f"{log_prefix} {msg}", flush=True)

    run_seeds = [1] if quick else list(seeds)
    run_qua = ["ACC"] if quick else list(qua_methods)
    run_tsa = ["QFY"] if quick else list(tsa_methods)
    run_clf = (classifiers[0],) if quick else classifiers
    run_exp = ["TOMS"] if quick else list(exp_types)

    if quick:
        log(
            "Tubular QUICK: seed=1, ExpType=TOMS, quantifier=ACC, TSA=QFY, "
            f"classifier={run_clf[0]!r}."
        )

    seed_tables = []
    total_steps = (
        len(run_seeds) * len(run_exp) * len(run_qua) * len(run_clf) * (len(run_tsa) + 1)
    )
    pbar = tqdm(total=total_steps, desc="Experiment (tubular)")
    columns = (
        "Dataset",
        "ExpType",
        "QuaMethod",
        "Classifier",
        "QFY",
        "MA",
        "KFMA",
    )

    for seed in run_seeds:
        idx = 0
        outputfile = pd.DataFrame({col: [] for col in columns})
        for exp_type in run_exp:
            for qua in run_qua:
                for mod in run_clf:
                    row = {
                        "Dataset": dataset_name,
                        "ExpType": exp_type,
                        "QuaMethod": qua,
                        "Classifier": mod,
                        "QFY": np.nan,
                        "MA": np.nan,
                        "KFMA": np.nan,
                    }
                    if exp_type == "original":
                        for tsa in run_tsa:
                            row[tsa] = tubular_single_experiment(
                                dataset,
                                mod,
                                qua,
                                tsa,
                                seed,
                                unified_window,
                                time_column,
                            )
                            pbar.update(1)
                    else:
                        row["QFY"] = tubular_toms_experiment(
                            cfg_toms, dataset, mod, qua, "QFY", seed
                        )
                        pbar.update(1)
                    outputfile.loc[idx] = row
                    idx += 1
        seed_tables.append(outputfile)
    pbar.close()

    metric_cols = ["QFY", "MA", "KFMA"]
    log(f"Averaging tubular results over {len(seed_tables)} seed(s) ...")
    stack = np.stack([tbl[metric_cols].to_numpy() for tbl in seed_tables])
    tot = np.nanmean(stack, axis=0)
    tot_res = seed_tables[0][["Dataset", "ExpType", "QuaMethod", "Classifier"]].copy()
    for i, m in enumerate(metric_cols):
        tot_res[m] = tot[:, i]

    tsa_for_best = run_tsa if quick else list(tsa_methods)
    best_m = annotate_best_method(tot_res[tsa_for_best], tsa_for_best)
    tot_res["best_method"] = best_m

    out_name = (
        "MAE_quanti_results_mean_tubular_quick.csv"
        if quick
        else "MAE_quanti_results_mean_tubular.csv"
    )
    out_path = config.QUANT_RESULTS_DIR / out_name
    tot_res.to_csv(out_path)
    log(f"Tubular results saved to {out_path}.")
