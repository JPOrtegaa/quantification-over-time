import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from . import HuggingFaceModel
from sklearn.metrics import accuracy_score, f1_score

_MODEL_CACHE = {}


def get_model(mod):
    if mod not in _MODEL_CACHE:
        _MODEL_CACHE[mod] = HuggingFaceModel.SentiAnalyzer(mod)
    return _MODEL_CACHE[mod]


# Preprocess text (username and link placeholders)
def analyzer(df_text, mod, classes):
    """
    :param df_text: a dataset with true labels
    :param mod: name(string) of a sentiment classifier or a trained classifier class
    :param classes: the name(string) of classes
    :return: [pred_labels, pred_scores, [acc, f1]]
    """

    # PART 1: text classification for twiiter datasets
    if type(mod) == str:
        df_x = df_text["text"].to_list()
        df_y = df_text["label"].reset_index(drop=True).astype(int)
        pred_labels = []
        pred_scores = []

        if len(classes) == 2:
            if mod == "vader":
                classifier = SentimentIntensityAnalyzer()

                for x in df_x:
                    res_x = classifier.polarity_scores(str(x))
                    if res_x["pos"] > res_x["neg"]:
                        pred_labels.append(1)
                    elif res_x["pos"] <= res_x["neg"]:
                        pred_labels.append(-1)
                    s = res_x["neg"] + res_x["pos"] + 2 * 1e-5
                    pred_scores.append(
                        [(res_x["neg"] + 1e-5) / s, (res_x["pos"] + 1e-5) / s]
                    )

            else:
                model = get_model(mod)
                scores_batch, labels_batch = model.batch_analyze(df_x)
                for scores, label in zip(scores_batch, labels_batch):
                    if scores[2] >= scores[0]:
                        pred_labels.append(1)
                    else:
                        pred_labels.append(-1)
                    s = scores[0] + scores[2]
                    pred_scores.append([scores[0] / s, scores[2] / s])

            pred_labels = pd.Series(pred_labels).astype(int)
            pred_scores = np.array(pred_scores)
            acc = (pred_labels == df_y).sum() / len(df_y)
            f1 = f1_score(df_y, pred_labels, average="macro")

            pred_labels = pd.DataFrame({"pred_y": pred_labels, "true_y": df_y})

            pred_scores = pd.DataFrame(
                {-1: pred_scores[:, 0], 1: pred_scores[:, 1], "true_y": df_y}
            )

        else:
            if mod == "vader":
                classifier = SentimentIntensityAnalyzer()

                for x in df_x:
                    res_x = classifier.polarity_scores(str(x))
                    if res_x["compound"] > 0.0:
                        pred_labels.append(1)
                    elif res_x["compound"] < 0.0:
                        pred_labels.append(-1)
                    else:
                        pred_labels.append(0)
                    pred_scores.append([res_x["neg"], res_x["neu"], res_x["pos"]])

            else:
                model = get_model(mod)
                scores_batch, labels_batch = model.batch_analyze(df_x)
                for scores, label in zip(scores_batch, labels_batch):
                    if label == "negative" or label == "Negative" or label == "LABEL_0":
                        pred_labels.append(-1)
                    elif (
                        label == "positive" or label == "Positive" or label == "LABEL_2"
                    ):
                        pred_labels.append(1)
                    else:
                        pred_labels.append(0)
                    pred_scores.append(scores)

            pred_labels = pd.Series(pred_labels).astype(int)
            pred_scores = np.array(pred_scores)
            acc = (pred_labels == df_y).sum() / len(df_y)
            acc_rate = ((pred_labels == df_y).sum() + 1e-4) / (
                (pred_labels != df_y).sum() + 1e-4
            )
            f1 = f1_score(df_y, pred_labels, average="macro")

            pred_labels = pd.DataFrame({"pred_y": pred_labels, "true_y": df_y})

            pred_scores = pd.DataFrame(
                {
                    -1: pred_scores[:, 0],
                    0: pred_scores[:, 1],
                    1: pred_scores[:, 2],
                    "true_y": df_y,
                }
            )

        return [pred_labels, pred_scores, [acc, f1]]

    # PART 2: numeral classification for mosquito datasets
    else:
        df_x = df_text.loc[:, ~df_text.columns.isin(["label"])]
        df_y = df_text["label"]

        pred_labels = mod.predict(df_x)
        pred_scores = mod.predict_proba(df_x)

        acc_rate = ((pred_labels == df_y).sum() + 1e-4) / (
            (pred_labels != df_y).sum() + 1e-4
        )
        acc = accuracy_score(df_y, pred_labels)
        f1 = f1_score(df_y, pred_labels, average="macro")

        pred_labels = pd.DataFrame({"pred_y": pred_labels, "true_y": df_y.to_numpy()})

        df_pred_scores = {}
        for cl in classes:
            index = mod.classes_.tolist().index(cl)
            df_pred_scores[cl] = pred_scores[:, index]
        df_pred_scores["true_y"] = df_y.to_numpy()
        df_pred_scores = pd.DataFrame(df_pred_scores)

        return [pred_labels, df_pred_scores, [acc, f1]]
