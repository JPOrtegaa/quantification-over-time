import config
from data_loading import loading
from classifier.scores_persist import (
    batch_split_window_keys,
    save_split_classifier_scores,
)
from utils.classifier_scores import compute_window_classifier_scores


def experiment(
    dataset,
    train_set_size,
    test_set_size,
    classifier,
    quantifier,
    random_state,
):
    loaded = loading(dataset)
    if len(loaded) == 3:
        data_dict, _prevalence_df, classes = loaded
    else:
        _training_data, data_dict, _prevalence_df, classes = loaded

    train_keys, test_keys = batch_split_window_keys(
        data_dict, train_set_size, test_set_size
    )
    keys_needed = train_keys + test_keys
    sub_dict = {k: data_dict[k] for k in keys_needed}

    all_scores = compute_window_classifier_scores(
        sub_dict,
        classifier,
        classes,
    )
    scores_train = {k: all_scores[k] for k in train_keys}
    scores_test = {k: all_scores[k] for k in test_keys}

    save_split_classifier_scores(
        config.CLASSIFIER_OUTPUT_DIR,
        dataset_name=str(dataset),
        classifier_model_id=classifier,
        classes=classes,
        scores_train=scores_train,
        scores_test=scores_test,
        train_keys=train_keys,
        test_keys=test_keys,
        train_set_size=train_set_size,
        test_set_size=test_set_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    experiment(
        dataset="global_covid19_tweets",
        train_set_size=15,
        test_set_size=None,
        classifier="amansolanki/autonlp-Tweet-Sentiment-Extraction-20114061",
        quantifier="DyS",
        random_state=1,
    )
