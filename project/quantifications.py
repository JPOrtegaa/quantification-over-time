import numpy as np
from quantifiers import DyS, DyS_Opt, ACC, GPAC, EDy
from sklearn.metrics import confusion_matrix
from classification import Classifying
import pandas as pd
from joblib import Parallel, delayed
from utils import mse, mae
import config


def _effective_loky_jobs(n_tasks):
    j = getattr(config, "TEST_CHUNK_LOKY_JOBS", -1)
    if j == 1 or n_tasks <= 1:
        return 1
    return j


def _analyze_test_chunks_parallel(test_set_dict, analyze_test, senti_model, classes, kind):
    """
    Executa analyze_test em cada chunk de teste em paralelo (backend loky).
    kind: "scores" -> dict[i] = DataFrame de scores; "labels" -> dict[i] = pred labels;
          "scores_arr" -> dict[i] = ndarray sem coluna true_y.
    """
    n = len(test_set_dict)
    if n == 0:
        return {}
    n_jobs = _effective_loky_jobs(n)
    if n_jobs == 1:
        out = {}
        for i in range(n):
            lab, sc, _ = analyze_test(test_set_dict[i], senti_model, classes)
            if kind == "scores":
                out[i] = sc
            elif kind == "labels":
                out[i] = lab
            elif kind == "scores_arr":
                out[i] = sc.iloc[:, :-1].to_numpy()
            else:
                raise ValueError(f"unknown kind {kind!r}")
        return out

    def work(i):
        lab, sc, _ = analyze_test(test_set_dict[i], senti_model, classes)
        if kind == "scores":
            return i, sc
        if kind == "labels":
            return i, lab
        if kind == "scores_arr":
            return i, sc.iloc[:, :-1].to_numpy()
        raise ValueError(f"unknown kind {kind!r}")

    pairs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(work)(i) for i in range(n)
    )
    return dict(pairs)


def _time_column_to_X(time_series):
    t = pd.to_datetime(time_series, errors="coerce")
    if t.isna().any():
        raise ValueError("Time column contains invalid or missing datetimes after parsing.")
    return (t.astype("int64").to_numpy(dtype=np.float64) / 1e9).reshape(-1, 1)


def _scores_from_regressor_time(df_text, classes, regressor, time_column):
    if time_column not in df_text.columns:
        raise KeyError(
            f"Time column {time_column!r} not found in dataframe columns: {list(df_text.columns)}"
        )
    X = _time_column_to_X(df_text[time_column])
    adjusted = regressor.predict(X)
    adjusted = np.asarray(adjusted, dtype=np.float64)
    if adjusted.ndim == 1:
        adjusted = adjusted.reshape(-1, 1)
    adjusted = np.clip(adjusted, 1e-8, None)
    row_sums = adjusted.sum(axis=1, keepdims=True)
    adjusted = adjusted / np.where(row_sums > 0, row_sums, 1.0)

    df_y = df_text["label"].reset_index(drop=True).astype(int)
    new_scores = pd.DataFrame({cl: adjusted[:, i] for i, cl in enumerate(classes)})
    new_scores["true_y"] = df_y

    class_cols = np.array(classes)
    idx = np.argmax(adjusted, axis=1)
    new_pred_y = class_cols[idx]
    new_labels = pd.DataFrame({"pred_y": new_pred_y, "true_y": df_y.to_numpy()})
    return new_labels, new_scores


def analyzer_with_regressor(df_text, mod, classes, regressor, time_column):
    _, pred_scores, metrics = Classifying.analyzer(df_text, mod, classes)
    new_labels, new_scores = _scores_from_regressor_time(
        df_text, classes, regressor, time_column
    )
    ty = pred_scores["true_y"].to_numpy()
    new_scores["true_y"] = ty
    new_labels["true_y"] = ty
    return new_labels, new_scores, metrics


def analyzer_regressor_test_only(df_text, mod, classes, regressor, time_column):
    """Scores só a partir de tempo + regressor (sem inferência HF). `mod` ignorado."""
    new_labels, new_scores = _scores_from_regressor_time(
        df_text, classes, regressor, time_column
    )
    return new_labels, new_scores, [0.0, 0.0]


def make_analyze_fn(regressor, time_column):
    def analyze_fn(df_text, mod, classes):
        return analyzer_with_regressor(df_text, mod, classes, regressor, time_column)

    return analyze_fn


def make_analyze_fn_test_regressor_only(regressor, time_column):
    def analyze_fn(df_text, mod, classes):
        return analyzer_regressor_test_only(df_text, mod, classes, regressor, time_column)

    return analyze_fn


def prepare_regressor_training_arrays(val_set, classifier, classes, time_column):
    _, pred_scores, _ = Classifying.analyzer(val_set, classifier, classes)
    if time_column not in val_set.columns:
        raise KeyError(
            f"time column {time_column!r} not found; available: {list(val_set.columns)}"
        )
    X = _time_column_to_X(val_set[time_column])
    Y = pred_scores[[cl for cl in classes]].to_numpy()
    return X, Y


def ACC_on_TSsets(
    val_set, test_set_dict, senti_model, classes, analyze_fn=None, analyze_fn_test=None
):
    analyze_val = analyze_fn or Classifying.analyzer
    analyze_test = analyze_fn_test or analyze_val
    val_y_res = analyze_val(val_set, senti_model, classes)
    val_y_ = val_y_res[0]

    tests_y_ = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "labels"
    )

    qtfied_distribution = []
    for i, cla in enumerate(classes):
        val_y_one = val_y_.copy()

        val_y_one.loc[val_y_one["true_y"] == cla, "true_y"] = "True"
        val_y_one.loc[val_y_one["true_y"].isin(classes), "true_y"] = "False"
        val_y_one.loc[val_y_one["pred_y"] == cla, "pred_y"] = "True"
        val_y_one.loc[val_y_one["pred_y"].isin(classes), "pred_y"] = "False"
        tn, fp, fn, tp = confusion_matrix(
            val_y_one["true_y"], val_y_one["pred_y"]
        ).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        qtfied_prevs_one = []
        for j in range(len(test_set_dict)):
            test_y_one = tests_y_[j]["pred_y"].copy()
            test_y_one[test_y_one == cla] = "True"
            test_y_one[test_y_one.isin(classes)] = "False"
            qua_prev = ACC(test_y_one, tpr, fpr)
            qtfied_prevs_one.append(qua_prev)
        qtfied_distribution.append(qtfied_prevs_one)

    qtfied_distribution = np.array(qtfied_distribution).T
    qtfied_distribution[np.all(qtfied_distribution == 0, axis=1)] = 1
    qtfied_distribution = qtfied_distribution / (
        np.sum(qtfied_distribution, axis=1).reshape(-1, 1)
    )
    return qtfied_distribution


def DyS_on_TSsets(
    val_set, test_set_dict, senti_model, classes, analyze_fn=None, analyze_fn_test=None
):
    analyze_val = analyze_fn or Classifying.analyzer
    analyze_test = analyze_fn_test or analyze_val
    val_y_res = analyze_val(val_set, senti_model, classes)
    val_score = val_y_res[1]

    tests_scores = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "scores"
    )

    qtfied_distribution = []
    for i, cla in enumerate(classes):
        val_score_one = val_score[val_score["true_y"] == cla][cla]
        val_score_rest = val_score[val_score["true_y"] != cla][cla]

        qtfied_prevs_one = []
        for j in range(len(test_set_dict)):
            test_score_one = tests_scores[j][cla]
            qua_prev = DyS(
                val_score_one, val_score_rest, test_score_one, measure="topsoe"
            )
            qtfied_prevs_one.append(qua_prev)
        qtfied_distribution.append(qtfied_prevs_one)

    qtfied_distribution = np.array(qtfied_distribution).T
    qtfied_distribution[np.all(qtfied_distribution == 0, axis=1)] = 1
    qtfied_distribution = qtfied_distribution / (
        np.sum(qtfied_distribution, axis=1).reshape(-1, 1)
    )

    return qtfied_distribution


def DyS_Opt_on_TSsets(
    val_set,
    test_set_dict,
    senti_model,
    classes,
    stride_ratio=0.05,
    analyze_fn=None,
    analyze_fn_test=None,
):
    analyze_val = analyze_fn or Classifying.analyzer
    analyze_test = analyze_fn_test or analyze_val
    val_y_res = analyze_val(val_set, senti_model, classes)
    val_score = val_y_res[1]

    tests_scores = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "scores"
    )

    qtfied_distribution = []
    for i, cla in enumerate(classes):
        val_score_one = val_score[val_score["true_y"] == cla][cla]
        val_score_rest = val_score[val_score["true_y"] != cla][cla]

        qtfied_prevs_one = []
        alpha_prev = None
        for j in range(len(test_set_dict)):
            if alpha_prev is None:
                current_left = 0.0
                current_right = 1.0
            else:
                current_left = max(0.0, alpha_prev - stride_ratio)
                current_right = min(1.0, alpha_prev + stride_ratio)
                
            test_score_one = tests_scores[j][cla]
            qua_prev = DyS_Opt(
                val_score_one, val_score_rest, test_score_one, measure="topsoe", left=current_left, right=current_right
            )
            alpha_prev = qua_prev
            qtfied_prevs_one.append(qua_prev)
        qtfied_distribution.append(qtfied_prevs_one)

    qtfied_distribution = np.array(qtfied_distribution).T
    qtfied_distribution[np.all(qtfied_distribution == 0, axis=1)] = 1
    qtfied_distribution = qtfied_distribution / (
        np.sum(qtfied_distribution, axis=1).reshape(-1, 1)
    )

    return qtfied_distribution


def GPAC_on_TSsets(
    val_set, test_set_dict, senti_model, classes, analyze_fn=None, analyze_fn_test=None
):
    analyze_val = analyze_fn or Classifying.analyzer
    analyze_test = analyze_fn_test or analyze_val
    val_res = analyze_val(val_set, senti_model, classes)
    val_scores = val_res[1].iloc[:, :-1].to_numpy()
    val_labels = val_res[0]

    tests_scores = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "scores_arr"
    )

    n_test = len(test_set_dict)
    nj = _effective_loky_jobs(n_test)
    val_y_np = val_labels["true_y"].to_numpy()
    if nj == 1:
        qtfied_distribution = [
            GPAC(val_scores, tests_scores[j], val_y_np, classes) for j in range(n_test)
        ]
    else:

        def gpac_j(j):
            return j, GPAC(val_scores, tests_scores[j], val_y_np, classes)

        pairs = Parallel(n_jobs=nj, backend="loky")(
            delayed(gpac_j)(j) for j in range(n_test)
        )
        pairs.sort(key=lambda x: x[0])
        qtfied_distribution = [p[1] for p in pairs]

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution


def EDy_on_TSsets(
    val_set, test_set_dict, senti_model, classes, analyze_fn=None, analyze_fn_test=None
):
    analyze_val = analyze_fn or Classifying.analyzer
    analyze_test = analyze_fn_test or analyze_val
    val_res = analyze_val(val_set, senti_model, classes)
    val_scores = val_res[1].iloc[:, :-1].to_numpy()
    val_labels = val_res[0]

    tests_scores = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "scores_arr"
    )

    n_test = len(test_set_dict)
    nj = _effective_loky_jobs(n_test)
    val_y_np = val_labels["true_y"].to_numpy()
    if nj == 1:
        qtfied_distribution = [
            EDy(val_scores, val_y_np, tests_scores[j], classes) for j in range(n_test)
        ]
    else:

        def edy_j(j):
            return j, EDy(val_scores, val_y_np, tests_scores[j], classes)

        pairs = Parallel(n_jobs=nj, backend="loky")(
            delayed(edy_j)(j) for j in range(n_test)
        )
        pairs.sort(key=lambda x: x[0])
        qtfied_distribution = [p[1] for p in pairs]

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution


def CC_on_TSsets(
    test_set_dict, senti_model, classes, analyze_fn=None, analyze_fn_test=None
):
    analyze_test = analyze_fn_test or analyze_fn or Classifying.analyzer
    tests_y_ = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "labels"
    )

    qtfied_distribution = []
    for j in range(len(test_set_dict)):
        qua_prev = []
        s = len(tests_y_[j])
        for c in classes:
            n = tests_y_[j][tests_y_[j]["pred_y"] == c]["pred_y"].count()
            qua_prev.append(n / s)

        qtfied_distribution.append(qua_prev)

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution


def PCC_on_TSsets(
    test_set_dict, senti_model, classes, analyze_fn=None, analyze_fn_test=None
):
    analyze_test = analyze_fn_test or analyze_fn or Classifying.analyzer
    tests_y_ = _analyze_test_chunks_parallel(
        test_set_dict, analyze_test, senti_model, classes, "scores"
    )

    qtfied_distribution = []
    for j in range(len(test_set_dict)):
        qua_prev = tests_y_[j].mean()[classes].to_numpy()
        qtfied_distribution.append(qua_prev)

    qtfied_distribution = np.array(qtfied_distribution)

    return qtfied_distribution

def getMAE_val_set(
    val_set,
    qua,
    mod,
    c,
    data,
    name,
    stride_ratio,
    random_seed,
    regressor=None,
    time_column=None,
):
    analyze_fn = None
    if regressor is not None:
        if time_column is None:
            raise ValueError("time_column is required when regressor is provided")
        analyze_fn = make_analyze_fn(regressor, time_column)

    subsamples_dict = {}
    subsamples_dsts = []
    for i in range(name[1]):
        subsamples_dict[i] = data[i]
        s = subsamples_dict[i].value_counts("label").sum()
        p = []
        for label in c:
            p.append(
                subsamples_dict[i][subsamples_dict[i]["label"] == label][
                    "label"
                ].count()
                / s
            )
        subsamples_dsts.append(p)

    subsamples_prevs = np.array(subsamples_dsts)

    val_MAE, val_MSE, sep_MAE, qtfd_dsts = None, None, None, None

    if qua == "DyS":
        qtfd_dsts = DyS_on_TSsets(val_set, subsamples_dict, mod, c, analyze_fn=analyze_fn)
    elif qua == "DyS-Opt":
        qtfd_dsts = DyS_Opt_on_TSsets(
            val_set, subsamples_dict, mod, c, stride_ratio=stride_ratio, analyze_fn=analyze_fn
        )
    elif qua == "ACC":
        qtfd_dsts = ACC_on_TSsets(val_set, subsamples_dict, mod, c, analyze_fn=analyze_fn)
    elif qua == "GPAC":
        qtfd_dsts = GPAC_on_TSsets(val_set, subsamples_dict, mod, c, analyze_fn=analyze_fn)
    elif qua == "EDy":
        qtfd_dsts = EDy_on_TSsets(val_set, subsamples_dict, mod, c, analyze_fn=analyze_fn)
    elif qua == "CC":
        qtfd_dsts = CC_on_TSsets(subsamples_dict, mod, c, analyze_fn=analyze_fn)
    elif qua == "PCC":
        qtfd_dsts = PCC_on_TSsets(subsamples_dict, mod, c, analyze_fn=analyze_fn)
    elif qua == "ReadMe2":
        val_preds_path = (
            config.README_IMPLEMENT_DIR
            / "data"
            / name[0]
            / f"seed{random_seed}"
            / "val_preds.csv"
        )
        qtfd_dsts = np.array(pd.read_csv(val_preds_path))

    sep_MAE = abs(subsamples_prevs - qtfd_dsts).sum(axis=0) / len(subsamples_dsts)
    val_MAE = mae(subsamples_prevs, qtfd_dsts)
    val_MSE = mse(subsamples_prevs, qtfd_dsts)
    return val_MAE, val_MSE, sep_MAE, qtfd_dsts


def qtfied_dists(
    valset,
    data_dict,
    dataname,
    qua,
    mod,
    c,
    stride_ratio,
    random_seed,
    regressor=None,
    time_column=None,
):
    analyze_fn = None
    analyze_fn_test = None
    if regressor is not None:
        if time_column is None:
            raise ValueError("time_column is required when regressor is provided")
        analyze_fn = make_analyze_fn(regressor, time_column)
        analyze_fn_test = make_analyze_fn_test_regressor_only(regressor, time_column)

    try:
        if regressor is not None:
            raise IOError("bypass disk cache when time regressor is used")
        results_file = (
            config.QUANT_RESULTS_DIR
            / f"_{dataname[0]}"
            / f"{qua}-{str(mod)[:6]}-{dataname[0]}-{dataname[1]}.csv"
        )
        quantified_dsts = (
            pd.read_csv(results_file).drop(labels=["Unnamed: 0"], axis=1).to_numpy()
        )
        if len(quantified_dsts) != len(data_dict):
            raise IOError("Stale results found (different number of windows)")

        quantified_dsts = np.nan_to_num(quantified_dsts, nan=1 / len(c))

    except IOError:
        quantified_dsts = 0
        if qua == "DyS":
            quantified_dsts = DyS_on_TSsets(
                valset,
                data_dict,
                mod,
                c,
                analyze_fn=analyze_fn,
                analyze_fn_test=analyze_fn_test,
            )
        elif qua == "DyS-Opt":
            quantified_dsts = DyS_Opt_on_TSsets(
                valset,
                data_dict,
                mod,
                c,
                stride_ratio=stride_ratio,
                analyze_fn=analyze_fn,
                analyze_fn_test=analyze_fn_test,
            )
        elif qua == "ACC":
            quantified_dsts = ACC_on_TSsets(
                valset,
                data_dict,
                mod,
                c,
                analyze_fn=analyze_fn,
                analyze_fn_test=analyze_fn_test,
            )
        elif qua == "GPAC":
            quantified_dsts = GPAC_on_TSsets(
                valset,
                data_dict,
                mod,
                c,
                analyze_fn=analyze_fn,
                analyze_fn_test=analyze_fn_test,
            )
        elif qua == "EDy":
            quantified_dsts = EDy_on_TSsets(
                valset,
                data_dict,
                mod,
                c,
                analyze_fn=analyze_fn,
                analyze_fn_test=analyze_fn_test,
            )
        elif qua == "CC":
            quantified_dsts = CC_on_TSsets(
                data_dict, mod, c, analyze_fn=analyze_fn, analyze_fn_test=analyze_fn_test
            )
        elif qua == "PCC":
            quantified_dsts = PCC_on_TSsets(
                data_dict, mod, c, analyze_fn=analyze_fn, analyze_fn_test=analyze_fn_test
            )
        elif qua == "ReadMe2":
            test_preds_path = (
                config.README_IMPLEMENT_DIR
                / "data"
                / dataname[0]
                / f"seed{random_seed}"
                / "test_preds.csv"
            )
            df_quantified_dsts = pd.read_csv(test_preds_path)
            quantified_dsts = np.array(df_quantified_dsts)

        quantified_dsts = np.nan_to_num(quantified_dsts, nan=1 / len(c))

        pd_quantified_dsts = {}
        for i, cls in enumerate(c):
            pd_quantified_dsts[cls] = quantified_dsts[:, i]
        pd_quantified_dsts = pd.DataFrame(quantified_dsts)
        dataset_folder = config.QUANT_RESULTS_DIR / f"_{dataname[0]}"
        if not dataset_folder.exists():
            dataset_folder.mkdir(parents=True, exist_ok=True)

        results_file = (
            dataset_folder / f"{qua}-{str(mod)[:6]}-{dataname[0]}-{dataname[1]}.csv"
        )
        if regressor is None:
            pd_quantified_dsts.to_csv(results_file)

    return quantified_dsts
