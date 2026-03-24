import argparse
import warnings

import config
import data_loading
import numpy as np
import pandas as pd
import quantifications as qfy
import utils
from regression import trainingModel as regression_trainingModel
from time_series_adjustment import KalmanMA, MovingAverage
from tqdm import tqdm
from utils import params_KFMA

warnings.filterwarnings("ignore")

seeds = [1, 2, 3]
DATASET = ("global_covid19_tweets", 15)
CLASSIFIERS = ["amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061"]
qua_methods = ["DyS", "DyS-Opt"]
TSA_methods = ["QFY", "MA", "KFMA"]
EXP_TYPES = ("original", "TOMS")
REGRESSOR_TIME_COLUMN = "TweetAt"
REGRESSOR_NAME = "LR"
unified_window = 4


def load_textual_series(dataset_name):
    return data_loading.loading(dataset_name)


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
    return inital_value, val_true, val_set, test_sets, test_dsts


def fit_time_classifier_output_regressor(val_set, classifier, classes, time_column, random_state):
    X, Y = qfy.prepare_regressor_training_arrays(val_set, classifier, classes, time_column)
    return regression_trainingModel.trainer(X, Y, REGRESSOR_NAME, random_state)


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

    modified_dsts = np.array(modified_dsts).T
    modified_dsts = modified_dsts / (np.sum(modified_dsts, axis=1).reshape(-1, 1))
    return utils.mae(test_dsts, modified_dsts)


def experiment(dataset, classifier, quantifier, tsa, random_state, exp_type):
    if exp_type == "TOMS" and tsa != "QFY":
        raise ValueError("TOMS only supports tsa='QFY'")

    ts_chunks, ts_prevalence, c, ts_info = load_textual_series(dataset[0])
    inital_value, val_true, val_set, test_sets, test_dsts = (
        compute_initial_window_and_split(dataset, ts_chunks, ts_prevalence)
    )

    regressor = None
    time_column = None
    if exp_type == "TOMS":
        regressor = fit_time_classifier_output_regressor(
            val_set, classifier, c, REGRESSOR_TIME_COLUMN, random_state
        )
        time_column = REGRESSOR_TIME_COLUMN

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

    return tsa_adjust_and_mae(
        tsa,
        quantified_dsts,
        test_dsts,
        inital_value,
        val_pred_dists,
        val_true,
        c,
        val_MSE,
    )


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


def run_textual_experiments():
    seed_tables = []
    total_steps = (
        len(seeds)
        * len(qua_methods)
        * len(CLASSIFIERS)
        * (len(TSA_methods) + 1)
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

    for seed in seeds:
        idx = 0
        outputfile = pd.DataFrame({col: [] for col in columns})
        for exp_type in EXP_TYPES:
            for qua in qua_methods:
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
                        for tsa in TSA_methods:
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
    pbar.close()

    metric_cols = ["QFY", "MA", "KFMA"]
    tot = aggregate_mean_over_seeds(seed_tables, metric_cols)
    tot_res = seed_tables[0][["Dataset", "ExpType", "QuaMethod", "Classifier"]].copy()
    for i, m in enumerate(metric_cols):
        tot_res[m] = tot[:, i]

    TSF = tot_res[metric_cols]
    best_m = annotate_best_method(TSF, TSA_methods)
    tot_res["best_method"] = best_m
    tot_res.to_csv(config.OUTPUT_DIR / "MAE_quanti_results_mean_global_textual.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=["global_textual"],
        default="global_textual",
        help="Run global Covid19 textual quantification experiment",
    )
    args = parser.parse_args()
    run_textual_experiments()
