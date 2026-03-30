"""
Core workflow for textual quantification experiments (time series + TOMS).

`master_textual_experiment` runs a single experiment; `run_textual_experiments_grid`
runs the full grid (seeds x quantifiers x ...).
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
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

warnings.filterwarnings("ignore")


@dataclass
class TextualExperimentConfig:
    """Hyperparameters and lists consumed by `run_textual_experiments_grid`."""

    seeds: Tuple[int, ...] = (1, 2, 3)
    val_length: int = 15
    max_test_chunks: int = 5000
    dataset_name: str = "global_covid19_tweets"
    classifiers: Tuple[str, ...] = field(
        default_factory=lambda: (
            "amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061",
        )
    )
    qua_methods: Tuple[str, ...] = ("DyS", "DyS-Opt")
    tsa_methods: Tuple[str, ...] = ("QFY", "MA", "KFMA")
    exp_types: Tuple[str, ...] = ("TOMS",)
    regressor_label: str = "TSMN"
    regressor_time_column: str = "TweetAt"
    # TOMS: "scalar" (epoch) or "week" (7-d weekday one-hot, Mon–Sun)
    regressor_time_encoding: str = "scalar"
    regressor_tsmn_kwargs: dict = field(
        default_factory=lambda: {"tsmn_mode": "linear", "tsmn_degree": 3}
    )
    unified_window: int = 4
    log_prefix: str = "[run_experiment]"

    @property
    def dataset(self) -> Tuple[str, int]:
        return (self.dataset_name, self.val_length)

    @property
    def regressor_log_path(self):
        return config.OUTPUT_REGRESSOR_DIR / "toms_regressor_log.txt"


def _log(cfg: TextualExperimentConfig, msg: str) -> None:
    print(f"{cfg.log_prefix} {msg}", flush=True)


def load_textual_series(cfg: TextualExperimentConfig, dataset_name: str):
    _log(cfg, f"Loading textual time series: {dataset_name!r} ...")
    out = data_loading.loading(dataset_name)
    _log(cfg, f"Data loaded ({dataset_name!r}).")
    return out


def truncate_time_series_chunks(
    cfg: TextualExperimentConfig,
    ts_chunks,
    ts_prevalence,
    val_length: int,
    max_test_chunks: int,
):
    """Keep only the first (val_length + max_test_chunks) chunks, aligned with ts_prevalence."""
    n_total = len(ts_chunks)
    n_keep = val_length + max_test_chunks
    if n_total <= n_keep:
        _log(
            cfg,
            f"Series not truncated: only {n_total} chunk(s) "
            f"(≤ val_length+max_test={n_keep}).",
        )
        return ts_chunks, ts_prevalence
    ts_new = {i: ts_chunks[i] for i in range(n_keep)}
    prev_new = ts_prevalence.iloc[:n_keep].copy().reset_index(drop=True)
    _log(
        cfg,
        f"Series truncated: {n_total} -> {n_keep} chunks "
        f"(val_length={val_length}, max_test_chunks={max_test_chunks}).",
    )
    return ts_new, prev_new


def compute_initial_window_and_split(
    cfg: TextualExperimentConfig, dataset, ts_chunks, ts_prevalence
):
    u = cfg.unified_window
    if dataset[1] < u:
        lf = 0
    else:
        lf = dataset[1] - u
    inital_value = ts_prevalence.iloc[lf : dataset[1], :].to_numpy()
    val_true = ts_prevalence[: dataset[1]].to_numpy()
    val_set, test_sets, test_dsts = utils.val_test_split(
        ts_chunks.copy(), ts_prevalence, dataset[1]
    )
    _log(
        cfg,
        "Train/val vs test split: "
        f"val_set rows={len(val_set)}, test chunks={len(test_sets)}, "
        f"val_length={dataset[1]}.",
    )
    return inital_value, val_true, val_set, test_sets, test_dsts


def run_validation_quantification(
    cfg: TextualExperimentConfig,
    val_set,
    quantifier,
    classifier,
    c,
    ts_chunks,
    dataset,
    ts_info,
    random_state,
    regressor=None,
    time_column=None,
):
    _log(
        cfg,
        "Validation step (getMAE_val_set): "
        f"quantifier={quantifier!r}, regressor={'yes' if regressor is not None else 'no'}.",
    )
    return qfy.getMAE_val_set(
        val_set,
        quantifier,
        classifier,
        c,
        ts_chunks,
        dataset,
        ts_info,
        random_seed=random_state,
        regressor=regressor,
        time_column=time_column,
    )


def run_test_quantification(
    cfg: TextualExperimentConfig,
    val_set,
    test_sets,
    dataset,
    quantifier,
    classifier,
    c,
    ts_info,
    random_state,
    regressor=None,
    time_column=None,
):
    extra = "."
    if regressor is not None:
        if isinstance(regressor, TOMSMultiRegressorBundle):
            extra = (
                " — TOMS: per chunk, quantifier calibrates on K rows of M(t) from regressors; "
                "chunk scores from classifier (HF). Val MAE uses M on each val window (base_wi=0)."
            )
        else:
            extra = " — test chunks scored only via time regressor (no HF)."
    _log(
        cfg,
        "Test step (qtfied_dists): "
        f"quantifier={quantifier!r}, regressor={'yes' if regressor is not None else 'no'}"
        f"{extra}",
    )
    return qfy.qtfied_dists(
        val_set,
        test_sets,
        dataset,
        quantifier,
        classifier,
        c,
        ts_info,
        random_seed=random_state,
        regressor=regressor,
        time_column=time_column,
    )


def tsa_adjust_compute_mae(
    cfg: TextualExperimentConfig,
    tsa,
    quantified_dsts,
    test_dsts,
    inital_value,
    val_pred_dists,
    val_true,
    c,
    val_MSE,
):
    qua_mae = utils.mae(test_dsts, quantified_dsts)
    if tsa == "QFY":
        _log(cfg, f"Temporal adjustment: QFY only (raw quantification MAE) = {qua_mae:.6f}.")
        return qua_mae

    _log(cfg, f"Temporal adjustment: applying {tsa!r} to quantified prevalences ...")
    modified_dsts = []
    val_init_value = np.empty((0, len(c)))
    u = cfg.unified_window
    validation = [val_init_value, val_pred_dists, val_true, c, u]
    if tsa == "MA":
        for index in range(len(c)):
            modified_prevs = MovingAverage(
                initial_value=inital_value[:, index],
                quantified_prevs=quantified_dsts[:, index],
                window=u,
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
                state_dim=u,
                q=q,
            )
            modified_dsts.append(modified_prevs)

    modified_dsts = np.array(modified_dsts).T
    modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))
    combi = utils.mae(test_dsts, modified_dsts)
    _log(cfg, f"Temporal adjustment {tsa!r} done (combined MAE) = {combi:.6f}.")
    return combi


def master_textual_experiment(
    cfg: TextualExperimentConfig,
    dataset,
    classifier,
    quantifier,
    tsa,
    random_state,
    exp_type: str,
) -> float:
    """
    Run one end-to-end experiment: load -> (TOMS?) -> per-window tables -> validation ->
    test quantification -> temporal adjustment -> MAE.
    """
    if exp_type == "TOMS" and tsa != "QFY":
        raise ValueError("TOMS only supports tsa='QFY'")

    _log(
        cfg,
        "=== experiment === "
        f"dataset={dataset[0]!r}, exp_type={exp_type!r}, "
        f"quantifier={quantifier!r}, tsa={tsa!r}, seed={random_state} "
        f"(classifier {str(classifier)[:50]}...)",
    )

    if (
        isinstance(dataset[0], str)
        and dataset[0].startswith("hotel")
        and cfg.regressor_time_column == "TweetAt"
    ):
        cfg.regressor_time_column = "date"
        _log(cfg, "Hotel dataset: regressor_time_column set to 'date' (full ISO timestamps → Unix).")

    ts_chunks, ts_prevalence, c, ts_info = load_textual_series(cfg, dataset[0])
    if dataset[0] == "global_covid19_tweets" or (
        isinstance(dataset[0], str) and dataset[0].startswith("hotel")
    ):
        ts_chunks, ts_prevalence = truncate_time_series_chunks(
            cfg, ts_chunks, ts_prevalence, dataset[1], cfg.max_test_chunks
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
        time_column=cfg.regressor_time_column,
    )
    _log(cfg, f"True per-window prevalence → {true_prev_path}")

    window_t = scalar_time_per_window(ts_chunks, cfg.regressor_time_column)
    window_row_features = build_window_row_features(
        ts_chunks,
        cfg.regressor_time_column,
        cfg.regressor_time_encoding,
        window_t,
    )
    mono = diagnose_scalar_time_monotonicity(ts_chunks, cfg.regressor_time_column)
    if mono["monotonic_non_decreasing"]:
        _log(
            cfg,
            f"t_window medians: non-decreasing in window order ({mono['n_windows']} windows).",
        )
    else:
        _log(
            cfg,
            "WARNING: t_window medians not non-decreasing — "
            f"first violations (up to 5): {mono['violations'][:5]}",
        )
    val_set = attach_window_ids(val_set, ts_chunks, dataset[1])

    clf_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(classifier))[:80]
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
            time_column=cfg.regressor_time_column,
        )
    finally:
        Classifying.HF_PHASE_HINT = None
    _log(cfg, f"Classifier window score table saved to {clf_out}.")

    regressor = None
    time_column: Optional[str] = None
    if exp_type == "TOMS":
        time_column = cfg.regressor_time_column
        log_path = cfg.regressor_log_path
        ts_run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(
                f"\n{'=' * 72}\n"
                f"TOMS multi-regressor run | utc={ts_run}\n"
                f"dataset={dataset[0]!r} quantifier={quantifier!r} seed={random_state}\n"
                f"classifier={str(classifier)[:120]}\n"
                f"window_t (first 5 windows): "
                f"{ {w: window_t[w] for w in sorted(window_t.keys())[:5]} }\n"
            )
        _log(
            cfg,
            "TOMS: training K class-conditional regressors "
            f"(time_encoding={cfg.regressor_time_encoding!r}; "
            "inference uses one row of features per window); "
            f"log file: {log_path}.",
        )
        reg_out = (
            config.OUTPUT_REGRESSOR_DIR
            / f"regressor_window_scores_{dataset[0]}_{quantifier}_seed{random_state}_{clf_slug}.csv"
        )
        train_reg_dir = config.OUTPUT_REGRESSOR_DIR / "train"
        test_reg_dir = config.OUTPUT_REGRESSOR_DIR / "test"
        Classifying.HF_PHASE_HINT = "TOMS train: val Y (HF)"
        try:
            span = qfy.date_span_label(val_set, cfg.regressor_time_column)
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
                cfg.regressor_time_column,
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
        _log(cfg, "TOMS: regressors trained.")
        write_bundle_window_scores_csv(
            ts_chunks, c, regressor, val_length=dataset[1], out_path=reg_out
        )
        _log(cfg, f"TOMS bundle table (M + row-mean scores) saved to {reg_out}.")
        _log(
            cfg,
            f"TOMS per-regressor train tables (K={len(c)} files) saved under {train_reg_dir} "
            f"(prefix {reg_out.stem!r}).",
        )
        write_toms_test_window_csvs(
            regressor,
            ts_chunks,
            c,
            val_length=dataset[1],
            out_dir=test_reg_dir,
            name_prefix=reg_out.stem,
        )
        _log(
            cfg,
            f"TOMS test window matrices (t + M) saved under {test_reg_dir} "
            f"(prefix {reg_out.stem!r}).",
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
            time_column=time_column,
        )
    finally:
        Classifying.HF_PHASE_HINT = None
    _log(cfg, "Validation finished; val metrics available.")

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
            time_column=time_column,
        )
    finally:
        Classifying.HF_PHASE_HINT = None
    _log(
        cfg,
        f"Test quantification finished (prevalence array shape: {quantified_dsts.shape}).",
    )

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
        time_column=cfg.regressor_time_column,
    )
    _log(cfg, f"Quantifier prevalence (test windows) → {quant_prev_path}")

    if exp_type == "TOMS" and regressor is not None:
        log_validation_and_test_matrices(
            regressor,
            val_set,
            ts_chunks,
            test_sets,
            classifier,
            c,
            cfg.regressor_time_column,
            dataset[1],
            cfg.regressor_log_path,
        )
        _log(cfg, f"Validation/test matrices appended to {cfg.regressor_log_path}.")

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
    _log(cfg, f"=== end experiment (final MAE this run) = {mae_out:.6f} ===")
    return mae_out


def aggregate_mean_over_seeds(seed_tables, metric_cols):
    stack = np.stack([tbl[metric_cols].to_numpy() for tbl in seed_tables])
    return np.nanmean(stack, axis=0)


def annotate_best_method(df, method_names):
    best_m = []
    for i in range(len(df)):
        row = df.iloc[i]
        mini = float("inf")
        m_num = -1
        for col_num, name in enumerate(method_names):
            v = row[name]
            if pd.notna(v) and v < mini:
                mini = v
                m_num = col_num
        best_m.append(method_names[m_num] if m_num >= 0 else "")
    return best_m


def run_textual_experiments_grid(cfg: TextualExperimentConfig, quick: bool = False):
    """
    Full grid: seeds x EXP_TYPES x quantifiers x classifiers x (TSA for original mode).
    Writes the aggregated CSV under `config.QUANT_RESULTS_DIR`.
    """
    run_seeds = [1] if quick else list(cfg.seeds)
    run_qua = ["ACC"] if quick else list(cfg.qua_methods)
    run_tsa = ["QFY"] if quick else list(cfg.tsa_methods)
    run_exp = list(cfg.exp_types)
    dataset = cfg.dataset

    if quick:
        _log(
            cfg,
            "@run_textual_experiments_grid · QUICK mode (VARIABLES.QUICK): "
            f"seeds={run_seeds}, qua={run_qua}, TSA(when original)={run_tsa}, "
            f"EXP_TYPES={run_exp} (smaller tqdm total).",
        )
    _log(
        cfg,
        "@run_textual_experiments_grid · starting global textual grid: "
        f"seeds={run_seeds}, EXP_TYPES={run_exp}, qua_methods={run_qua}, "
        f"TSA_methods={run_tsa} (TOMS uses QFY only); "
        f"parallel_test_chunks={config.TEST_CHUNK_LOKY_JOBS}, "
        f"HF_INFERENCE_BATCH_SIZE={config.HF_INFERENCE_BATCH_SIZE}, "
        f"SKLEARN_N_JOBS={config.SKLEARN_N_JOBS}.",
    )
    seed_tables = []
    total_steps = (
        len(run_seeds) * len(run_qua) * len(cfg.classifiers) * (len(run_tsa) + 1)
    )
    pbar = tqdm(total=total_steps, desc="Experiment (global textual)")
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
        _log(cfg, f"--- New round: seed={seed} ---")
        idx = 0
        outputfile = pd.DataFrame({col: [] for col in columns})
        for exp_type in run_exp:
            for qua in run_qua:
                for clf in cfg.classifiers:
                    row = {
                        "Dataset": dataset[0],
                        "ExpType": exp_type,
                        "QuaMethod": qua,
                        "Classifier": clf,
                        "QFY": np.nan,
                        "MA": np.nan,
                        "KFMA": np.nan,
                    }
                    if exp_type == "original":
                        for tsa in run_tsa:
                            row[tsa] = master_textual_experiment(
                                cfg, dataset, clf, qua, tsa, seed, exp_type
                            )
                            pbar.update(1)
                    else:
                        row["QFY"] = master_textual_experiment(
                            cfg, dataset, clf, qua, "QFY", seed, exp_type
                        )
                        pbar.update(1)
                    outputfile.loc[idx] = row
                    idx += 1

        seed_tables.append(outputfile)
        _log(cfg, f"Seed {seed}: table for this seed has {len(outputfile)} row(s).")
    pbar.close()

    metric_cols = ["QFY", "MA", "KFMA"]
    _log(cfg, f"Averaging over {len(seed_tables)} seed(s) ...")
    tot = aggregate_mean_over_seeds(seed_tables, metric_cols)
    tot_res = seed_tables[0][["Dataset", "ExpType", "QuaMethod", "Classifier"]].copy()
    for i, m in enumerate(metric_cols):
        tot_res[m] = tot[:, i]

    TSF = tot_res[metric_cols]
    tsa_for_best = run_tsa if quick else list(cfg.tsa_methods)
    best_m = annotate_best_method(TSF, tsa_for_best)
    tot_res["best_method"] = best_m
    out_name = (
        "MAE_quanti_results_mean_global_textual_quick.csv"
        if quick
        else "MAE_quanti_results_mean_global_textual.csv"
    )
    out_path = config.QUANT_RESULTS_DIR / out_name
    tot_res.to_csv(out_path)
    _log(cfg, f"Results saved to {out_path}.")
