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
# Janelas temporais iniciais usadas como validação (treino do regressor / referência quantificação).
VAL_LENGTH = 15
# Após o split, no máximo estes chunks entram como teste temporal (corta a série no fim).
MAX_TEST_CHUNKS = 10
DATASET = ("global_covid19_tweets", VAL_LENGTH)
CLASSIFIERS = ["amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061"]
qua_methods = ["DyS", "DyS-Opt"]
TSA_methods = ["QFY", "MA", "KFMA"]
#EXP_TYPES = ("original", "TOMS")
EXP_TYPES = ("TOMS")
REGRESSOR_TIME_COLUMN = "TweetAt"
REGRESSOR_NAME = "TSMN"
REGRESSOR_TSMN_KWARGS = {"tsmn_mode": "cyclic", "tsmn_degree": 3}
unified_window = 4

_LOG_PREFIX = "[run_experiment]"


def _log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", flush=True)


def load_textual_series(dataset_name):
    _log(f"Carregando série textual: {dataset_name!r} …")
    out = data_loading.loading(dataset_name)
    _log(f"Dados carregados ({dataset_name!r}).")
    return out


def truncate_time_series_chunks(ts_chunks, ts_prevalence, val_length, max_test_chunks):
    """
    Mantém só os primeiros (val_length + max_test_chunks) chunks da série,
    alinhado às linhas de ts_prevalence.
    """
    n_total = len(ts_chunks)
    n_keep = val_length + max_test_chunks
    if n_total <= n_keep:
        _log(
            f"Série não truncada: só há {n_total} chunks "
            f"(≤ val_length+max_test={n_keep})."
        )
        return ts_chunks, ts_prevalence
    ts_new = {i: ts_chunks[i] for i in range(n_keep)}
    prev_new = ts_prevalence.iloc[:n_keep].copy().reset_index(drop=True)
    _log(
        f"Série truncada: {n_total} → {n_keep} chunks "
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
        "Split val/test: "
        f"len(val_set)={len(val_set)} linhas, "
        f"chunks de teste={len(test_sets)}, "
        f"val_length={dataset[1]}."
    )
    return inital_value, val_true, val_set, test_sets, test_dsts


def fit_time_classifier_output_regressor(val_set, classifier, classes, time_column, random_state):
    _log(
        "TOMS: preparando (X=tempo, Y=scores) e treinando regressor "
        f"({REGRESSOR_NAME!r}, seed={random_state}) em {len(val_set)} linhas …"
    )
    X, Y = qfy.prepare_regressor_training_arrays(val_set, classifier, classes, time_column)
    _log(f"TOMS: shapes treino regressor X={X.shape}, Y={Y.shape}.")
    reg = regression_trainingModel.trainer(
        X, Y, REGRESSOR_NAME, random_state, **REGRESSOR_TSMN_KWARGS
    )
    _log("TOMS: regressor treinado.")
    return reg


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
        "Etapa validação (getMAE_val_set): "
        f"qua={quantifier!r}, regressor={'sim' if regressor is not None else 'não'}."
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
    _log(
        "Etapa teste (qtfied_dists): "
        f"qua={quantifier!r}, regressor={'sim' if regressor is not None else 'não'}"
        + (
            " — scores dos chunks de teste só via tempo+regressor (sem HF)."
            if regressor is not None
            else "."
        )
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
        _log(f"Ajuste temporal: QFY somente (MAE quantificação pura) = {qua_mae:.6f}.")
        return qua_mae

    _log(f"Ajuste temporal: aplicando {tsa!r} sobre prevalências quantificadas …")
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
    _log(f"Ajuste temporal {tsa!r} concluído (MAE combinado) = {combi:.6f}.")
    return combi


def experiment(dataset, classifier, quantifier, tsa, random_state, exp_type):
    if exp_type == "TOMS" and tsa != "QFY":
        raise ValueError("TOMS only supports tsa='QFY'")

    _log(
        "=== experimento === "
        f"dataset={dataset[0]!r}, exp_type={exp_type!r}, "
        f"qua={quantifier!r}, tsa={tsa!r}, seed={random_state} "
        f"(classificador {str(classifier)[:50]}…)"
    )

    ts_chunks, ts_prevalence, c, ts_info = load_textual_series(dataset[0])
    if dataset[0] == "global_covid19_tweets":
        ts_chunks, ts_prevalence = truncate_time_series_chunks(
            ts_chunks, ts_prevalence, dataset[1], MAX_TEST_CHUNKS
        )
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
    _log("Validação concluída; métricas val obtidas.")

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
    _log(f"Quantificação no teste concluída (formato prevalências: {quantified_dsts.shape}).")

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
    _log(f"=== fim experimento (MAE final deste run) = {mae_out:.6f} ===")
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


def run_textual_experiments(quick: bool = False):
    run_seeds = [1] if quick else seeds
    run_qua = ["DyS"] if quick else qua_methods
    run_tsa = ["QFY"] if quick else TSA_methods
    run_exp = EXP_TYPES

    if quick:
        _log(
            "MODO RÁPIDO (--quick): "
            f"seeds={run_seeds}, qua={run_qua}, TSA(original)={run_tsa}, "
            f"EXP_TYPES={run_exp} (tqdm total menor)."
        )
    _log(
        "Iniciando grid global textual: "
        f"seeds={run_seeds}, EXP_TYPES={run_exp}, qua_methods={run_qua}, "
        f"TSA_methods={run_tsa} (TOMS usa só QFY); "
        f"loky_jobs={config.TEST_CHUNK_LOKY_JOBS}, "
        f"HF_batch={config.HF_INFERENCE_BATCH_SIZE}, "
        f"sklearn_n_jobs={config.SKLEARN_N_JOBS}."
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
        _log(f"--- Nova rodada: seed={seed} ---")
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
        _log(f"Seed {seed}: tabela desta semente com {len(outputfile)} linhas.")
    pbar.close()

    metric_cols = ["QFY", "MA", "KFMA"]
    _log(f"Agregando média sobre {len(seed_tables)} seeds …")
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
    _log(f"Resultados salvos em {out_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        choices=["global_textual"],
        default="global_textual",
        help="Run global Covid19 textual quantification experiment",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Smoke test: 1 seed, só DyS, só QFY no original (+ TOMS com QFY). "
            "Salva em MAE_quanti_results_mean_global_textual_quick.csv"
        ),
    )
    args = parser.parse_args()
    _log(f"__main__: args.run={args.run!r}, quick={args.quick}")
    run_textual_experiments(quick=args.quick)
    _log("Execução principal concluída.")
