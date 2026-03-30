import argparse
import re
import warnings
from datetime import datetime, timezone

import config
from classification import Classifying
from classification import trainingModel
import data_loading_old as data_loading
import numpy as np
import pandas as pd
import quantifications as qfy
import utils
from regression.toms_multi_regressor import (
    TOMSMultiRegressorBundle,
    attach_window_ids,
    fit_toms_multi_regressors,
    log_validation_and_test_matrices,
    scalar_time_per_window,
    write_bundle_window_scores_csv,
)
from time_series_adjustment import KalmanMA, MovingAverage
from tqdm import tqdm
from utils import params_KFMA

warnings.filterwarnings("ignore")

seeds = [1]
# First VAL_LENGTH time windows: validation (regressor training + quantification reference).
VAL_LENGTH = 15
# After the split, at most this many additional chunks are kept as temporal test (truncates tail).
MAX_TEST_CHUNKS = 5000

SENTIMENT_TIMESTAMP_COLS = {
    "global_covid19_tweets": "TweetAt",
    "nepali_dataset_eng": "Datetime",
    "Apple-Twitter-Sentiment-DFE": "date",
}

DATASET = ("global_covid19_tweets", VAL_LENGTH)
# DATASET = ("nepali_dataset_eng", VAL_LENGTH)
# DATASET = ("Apple-Twitter-Sentiment-DFE", VAL_LENGTH)

TABULAR_DATASETS = [
    ("bike", 55),
    ("energy", 20),
    ("news", 36),
]
TABULAR_CLASSIFIERS = ["LR", "RF"]

CLASSIFIERS = ["amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061"]
qua_methods = ["DyS"]
TSA_methods = ["QFY"]
EXP_TYPES = ["original"]
# EXP_TYPES = ["TOMS"]
REGRESSOR_TIME_COLUMN = SENTIMENT_TIMESTAMP_COLS.get(DATASET[0], "TweetAt")
REGRESSOR_NAME = "TSMN"
REGRESSOR_TSMN_KWARGS = {"tsmn_mode": "polynomial", "tsmn_degree": 3}
unified_window = 4
REGRESSOR_LOG_PATH = config.PROJECT_ROOT / "regressor" / "toms_regressor_log.txt"

_LOG_PREFIX = "[run_experiment]"

def textual_time_column(dataset_name: str) -> str:
    """Timestamp column in chunk DataFrames for regressors, classifier tables, and CM."""
    if dataset_name in SENTIMENT_TIMESTAMP_COLS:
        return SENTIMENT_TIMESTAMP_COLS[dataset_name]
    if isinstance(dataset_name, str) and dataset_name.startswith("hotel_neutral_"):
        return "date"
    return "TweetAt"


def hotel_neutral_dataset_tuples():
    """(dataset_name, val_length) for each hotel*.csv under hotel-datasets-neutral."""
    d = config.DATA_DIR / "hotel-datasets-neutral"
    if not d.is_dir():
        return []
    return [
        (f"hotel_neutral_{path.stem}", VAL_LENGTH)
        for path in sorted(d.glob("hotel*.csv"))
    ]

def _log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", flush=True)


def load_textual_series(dataset_name):
    _log(f"Loading textual time series: {dataset_name!r} …")
    out = data_loading.loading(dataset_name)
    _log(f"Data loaded ({dataset_name!r}).")
    return out


def truncate_time_series_chunks(ts_chunks, ts_prevalence, val_length, max_test_chunks):
    """
    Keep only the first (val_length + max_test_chunks) chunks, aligned with ts_prevalence rows.
    """
    n_total = len(ts_chunks)
    n_keep = val_length + max_test_chunks
    if n_total <= n_keep:
        _log(
            f"Series not truncated: only {n_total} chunk(s) "
            f"(≤ val_length+max_test={n_keep})."
        )
        return ts_chunks, ts_prevalence
    ts_new = {i: ts_chunks[i] for i in range(n_keep)}
    prev_new = ts_prevalence.iloc[:n_keep].copy().reset_index(drop=True)
    _log(
        f"Series truncated: {n_total} → {n_keep} chunks "
        f"(val_length={val_length}, max_test_chunks={max_test_chunks})."
    )
    return ts_new, prev_new


def compute_initial_window_and_split(dataset, ts_chunks, ts_prevalence):
    if dataset[1] < unified_window:
        lf = 0
    else:
        lf = dataset[1] - unified_window
    inital_value = ts_prevalence.iloc[lf : dataset[1], :].to_numpy()
    val_true = ts_prevalence[: dataset[1]].to_numpy()
    val_set, test_sets, test_dsts = utils.val_test_split(
        ts_chunks.copy(), ts_prevalence, dataset[1]
    )
    _log(
        "Train/val vs test split: "
        f"val_set rows={len(val_set)}, test chunks={len(test_sets)}, "
        f"val_length={dataset[1]}."
    )
    return inital_value, val_true, val_set, test_sets, test_dsts


def run_validation_quantification(
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
        "Validation step (getMAE_val_set): "
        f"quantifier={quantifier!r}, regressor={'yes' if regressor is not None else 'no'}."
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
                " — val: scores from row-mean of M (K time regressors); "
                "test: classifier (HF) scores only."
            )
        else:
            extra = " — test chunks scored only via time regressor (no HF)."
    _log(
        "Test step (qtfied_dists): "
        f"quantifier={quantifier!r}, regressor={'yes' if regressor is not None else 'no'}"
        f"{extra}"
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


def tsa_adjust_and_mae(
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
        _log(f"Temporal adjustment: QFY only (raw quantification MAE) = {qua_mae:.6f}.")
        return qua_mae

    _log(f"Temporal adjustment: applying {tsa!r} to quantified prevalences …")
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

    modified_dsts = np.array(modified_dsts).T
    modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))
    combi = utils.mae(test_dsts, modified_dsts)
    _log(f"Temporal adjustment {tsa!r} done (combined MAE) = {combi:.6f}.")
    return combi


def experiment(dataset, classifier, quantifier, tsa, random_state, exp_type):
    if exp_type == "TOMS" and tsa != "QFY":
        raise ValueError("TOMS only supports tsa='QFY'")

    _log(
        "=== experiment === "
        f"dataset={dataset[0]!r}, exp_type={exp_type!r}, "
        f"quantifier={quantifier!r}, tsa={tsa!r}, seed={random_state} "
        f"(classifier {str(classifier)[:50]}…)"
    )

    ts_chunks, ts_prevalence, c, ts_info = load_textual_series(dataset[0])
    ts_chunks, ts_prevalence = truncate_time_series_chunks(
        ts_chunks, ts_prevalence, dataset[1], MAX_TEST_CHUNKS
    )

    time_col = textual_time_column(dataset[0])

    # --- Per-chunk confusion matrices (before val/test split) ---
    ts_col = SENTIMENT_TIMESTAMP_COLS.get(dataset[0])
    if ts_col is None and str(dataset[0]).startswith("hotel_neutral_"):
        ts_col = "date"
    if ts_col is not None:
        cm_df = qfy.compute_chunk_confusion_matrices(
            ts_chunks, classifier, c, timestamp_col=ts_col
        )
        if str(dataset[0]).startswith("hotel_neutral_"):
            cm_path = config.OUTPUT_DIR / f"confusion_matrices_{dataset[0]}_monthly.csv"
            cm_df = cm_df.copy()
            cm_df["timestamp"] = pd.to_datetime(
                cm_df["timestamp"], utc=True, errors="coerce"
            ).dt.strftime("%Y-%m")
            _log(
                "Hotel neutral: UTC calendar-month chunks; confusion matrix `timestamp` column is YYYY-MM."
            )
        else:
            cm_path = config.OUTPUT_DIR / f"confusion_matrices_{dataset[0]}.csv"
        cm_df.to_csv(cm_path, index=False)
        _log(f"Confusion matrices saved to {cm_path}")

    inital_value, val_true, val_set, test_sets, test_dsts = (
        compute_initial_window_and_split(dataset, ts_chunks, ts_prevalence)
    )

    window_t = scalar_time_per_window(ts_chunks, time_col)
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
            time_column=time_col,
        )
    finally:
        Classifying.HF_PHASE_HINT = None
    _log(f"Classifier window score table saved to {clf_out}.")

    regressor = None
    time_column = None
    if exp_type == "TOMS":
        time_column = time_col
        ts_run = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        REGRESSOR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGRESSOR_LOG_PATH, "a", encoding="utf-8") as lf:
            lf.write(
                f"\n{'=' * 72}\n"
                f"TOMS multi-regressor run | utc={ts_run}\n"
                f"dataset={dataset[0]!r} quantifier={quantifier!r} seed={random_state}\n"
                f"classifier={str(classifier)[:120]}\n"
                f"window_t (first 5 windows): "
                f"{ {w: window_t[w] for w in sorted(window_t.keys())[:5]} }\n"
            )
        _log(
            "TOMS: training K class-conditional time regressors (scalar t per window); "
            f"log file: {REGRESSOR_LOG_PATH}."
        )
        Classifying.HF_PHASE_HINT = "TOMS train: val Y (HF)"
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
                REGRESSOR_NAME,
                random_state,
                REGRESSOR_LOG_PATH,
                **REGRESSOR_TSMN_KWARGS,
            )
        finally:
            Classifying.HF_PHASE_HINT = None
        _log("TOMS: regressors trained.")
        reg_out = (
            config.OUTPUT_REGRESSOR_DIR
            / f"regressor_window_scores_{dataset[0]}_{quantifier}_seed{random_state}_{clf_slug}.csv"
        )
        write_bundle_window_scores_csv(
            ts_chunks, c, regressor, val_length=dataset[1], out_path=reg_out
        )
        _log(f"TOMS bundle table (M + row-mean scores) saved to {reg_out}.")

    Classifying.HF_PHASE_HINT = f"val MAE | {quantifier}"
    try:
        val_MAE, val_MSE, sep_mae, val_pred_dists = run_validation_quantification(
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
    _log("Validation finished; val metrics available.")

    Classifying.HF_PHASE_HINT = f"test qtfy | {quantifier}"
    try:
        quantified_dsts = run_test_quantification(
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
    _log(f"Test quantification finished (prevalence array shape: {quantified_dsts.shape}).")

    if exp_type == "TOMS" and regressor is not None:
        log_validation_and_test_matrices(
            regressor,
            val_set,
            ts_chunks,
            test_sets,
            classifier,
            c,
            time_col,
            dataset[1],
            REGRESSOR_LOG_PATH,
        )
        _log(f"Validation/test matrices appended to {REGRESSOR_LOG_PATH}.")

    mae_out = tsa_adjust_and_mae(
        tsa,
        quantified_dsts,
        test_dsts,
        inital_value,
        val_pred_dists,
        val_true,
        c,
        val_MSE,
    )
    _log(f"=== end experiment (final MAE this run) = {mae_out:.6f} ===")
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


def run_tabular_confusion_matrices():
    """Train LR and RF on each tabular dataset and save per-chunk confusion matrices."""
    for dataset in TABULAR_DATASETS:
        dataset_name = dataset[0]
        _log(f"Loading tabular dataset: {dataset_name!r} …")
        training_set, ts_chunks, ts_prevalence, c, ts_info = data_loading.loading(
            dataset_name
        )
        train_x = training_set.loc[:, ~training_set.columns.isin(["label"])]
        train_y = training_set["label"]

        for clf_name in TABULAR_CLASSIFIERS:
            _log(f"Training {clf_name} on {dataset_name!r} …")
            fitted_clf = trainingModel.trainer(train_x, train_y, clf_name, seed=1)

            _log(f"Computing per-chunk confusion matrices ({clf_name}) …")
            cm_df = qfy.compute_chunk_confusion_matrices(
                ts_chunks, fitted_clf, c, timestamp_col=None
            )
            cm_path = config.OUTPUT_DIR / f"confusion_matrices_{dataset_name}_{clf_name}.csv"
            cm_df.to_csv(cm_path, index=False)
            _log(f"Saved {cm_path}")

    _log("Tabular confusion matrix extraction finished.")


def run_textual_experiments(quick: bool = False):
    run_seeds = [1] if quick else seeds
    run_qua = ["DyS"] if quick else qua_methods
    run_tsa = ["QFY"] if quick else TSA_methods
    run_exp = EXP_TYPES

    if quick:
        _log(
            "QUICK mode (--quick): "
            f"seeds={run_seeds}, qua={run_qua}, TSA(when original)={run_tsa}, "
            f"EXP_TYPES={run_exp} (smaller tqdm total)."
        )
    _log(
        "Starting global textual grid: "
        f"seeds={run_seeds}, EXP_TYPES={run_exp}, qua_methods={run_qua}, "
        f"TSA_methods={run_tsa} (TOMS uses QFY only); "
        f"parallel_test_chunks={config.TEST_CHUNK_LOKY_JOBS}, "
        f"HF_INFERENCE_BATCH_SIZE={config.HF_INFERENCE_BATCH_SIZE}, "
        f"SKLEARN_N_JOBS={config.SKLEARN_N_JOBS}."
    )
    seed_tables = []
    total_steps = (
        len(run_seeds)
        * len(run_qua)
        * len(CLASSIFIERS)
        * (len(run_tsa) + 1)
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
        _log(f"--- New round: seed={seed} ---")
        idx = 0
        outputfile = pd.DataFrame({col: [] for col in columns})
        for exp_type in run_exp:
            for qua in run_qua:
                for clf in CLASSIFIERS:
                    row = {
                        "Dataset": DATASET[0],
                        "ExpType": exp_type,
                        "QuaMethod": qua,
                        "Classifier": clf,
                        "QFY": np.nan,
                        "MA": np.nan,
                        "KFMA": np.nan,
                    }
                    if exp_type == "original":
                        for tsa in run_tsa:
                            row[tsa] = experiment(
                                DATASET, clf, qua, tsa, seed, exp_type
                            )
                            pbar.update(1)
                    else:
                        row["QFY"] = experiment(
                            DATASET, clf, qua, "QFY", seed, exp_type
                        )
                        pbar.update(1)
                    outputfile.loc[idx] = row
                    idx += 1

        seed_tables.append(outputfile)
        _log(f"Seed {seed}: table for this seed has {len(outputfile)} row(s).")
    pbar.close()

    metric_cols = ["QFY", "MA", "KFMA"]
    _log(f"Averaging over {len(seed_tables)} seed(s) …")
    tot = aggregate_mean_over_seeds(seed_tables, metric_cols)
    tot_res = seed_tables[0][["Dataset", "ExpType", "QuaMethod", "Classifier"]].copy()
    for i, m in enumerate(metric_cols):
        tot_res[m] = tot[:, i]

    TSF = tot_res[metric_cols]
    best_m = annotate_best_method(TSF, run_tsa if quick else TSA_methods)
    tot_res["best_method"] = best_m
    out_name = (
        "MAE_quanti_results_mean_global_textual_quick.csv"
        if quick
        else "MAE_quanti_results_mean_global_textual.csv"
    )
    out_path = config.OUTPUT_DIR / out_name
    tot_res.to_csv(out_path)
    _log(f"Results saved to {out_path}.")


def run_hotel_neutral_experiments(quick: bool = False):
    """Quantification grid on every CSV under hotel-datasets-neutral (same HF model, VAL_LENGTH, unified_window)."""
    hotel_datasets = hotel_neutral_dataset_tuples()
    if not hotel_datasets:
        _log(
            "No hotel neutral datasets found "
            f"(expected {config.DATA_DIR / 'hotel-datasets-neutral'} with hotel*.csv). Skipping."
        )
        return

    run_seeds = [1] if quick else seeds
    run_qua = ["DyS"] if quick else qua_methods
    run_tsa = ["QFY"] if quick else TSA_methods
    run_exp = EXP_TYPES

    per_exp_steps = sum(len(run_tsa) if e == "original" else 1 for e in run_exp)
    total_steps = (
        len(run_seeds)
        * len(hotel_datasets)
        * len(run_qua)
        * len(CLASSIFIERS)
        * per_exp_steps
    )
    _log(
        f"Starting hotel neutral grid: {len(hotel_datasets)} dataset(s), "
        f"seeds={run_seeds}, EXP_TYPES={run_exp}, qua_methods={run_qua}, "
        f"TSA_methods={run_tsa}; total_steps={total_steps}."
    )

    seed_tables = []
    pbar = tqdm(total=total_steps, desc="Experiment (hotel neutral)")
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
        _log(f"--- Hotel neutral: seed={seed} ---")
        idx = 0
        outputfile = pd.DataFrame({col: [] for col in columns})
        for ds in hotel_datasets:
            for exp_type in run_exp:
                for qua in run_qua:
                    for clf in CLASSIFIERS:
                        row = {
                            "Dataset": ds[0],
                            "ExpType": exp_type,
                            "QuaMethod": qua,
                            "Classifier": clf,
                            "QFY": np.nan,
                            "MA": np.nan,
                            "KFMA": np.nan,
                        }
                        if exp_type == "original":
                            for tsa in run_tsa:
                                row[tsa] = experiment(
                                    ds, clf, qua, tsa, seed, exp_type
                                )
                                pbar.update(1)
                        else:
                            row["QFY"] = experiment(
                                ds, clf, qua, "QFY", seed, exp_type
                            )
                            pbar.update(1)
                        outputfile.loc[idx] = row
                        idx += 1

        seed_tables.append(outputfile)
        _log(f"Seed {seed}: {len(outputfile)} row(s).")
    pbar.close()

    metric_cols = ["QFY", "MA", "KFMA"]
    tot = aggregate_mean_over_seeds(seed_tables, metric_cols)
    tot_res = seed_tables[0][["Dataset", "ExpType", "QuaMethod", "Classifier"]].copy()
    for i, m in enumerate(metric_cols):
        tot_res[m] = tot[:, i]

    TSF = tot_res[metric_cols]
    best_m = annotate_best_method(TSF, run_tsa if quick else TSA_methods)
    tot_res["best_method"] = best_m
    out_name = (
        "MAE_quanti_results_mean_hotel_neutral_quick.csv"
        if quick
        else "MAE_quanti_results_mean_hotel_neutral.csv"
    )
    out_path = config.OUTPUT_DIR / out_name
    tot_res.to_csv(out_path)
    _log(f"Hotel neutral results saved to {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=["global_textual", "tabular_cm", "hotel_neutral"],
        default="global_textual",
        help=(
            "global_textual: sentiment quantification experiment; "
            "tabular_cm: extract confusion matrices for bike/energy/news with LR and RF"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke test: 1 seed, DyS only, QFY only for original (+ TOMS with QFY). "
            "Writes MAE_quanti_results_mean_global_textual_quick.csv"
        ),
    )
    args = parser.parse_args()
    _log(f"__main__: args.run={args.run!r}, quick={args.quick}")
    if args.run == "tabular_cm":
        run_tabular_confusion_matrices()
    elif args.run == "hotel_neutral":
        run_hotel_neutral_experiments(quick=args.quick)
    else:
        run_textual_experiments(quick=args.quick)
    _log("Main run finished.")
