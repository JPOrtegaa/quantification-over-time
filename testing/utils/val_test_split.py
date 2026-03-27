"""
Train/validation vs time-series test split for quantification experiments.
"""

import pandas as pd


def val_test_split(data, prevalences, val_length):
    if val_length == 0:
        return pd.DataFrame(), data, prevalences

    validation_set = data[0]
    del data[0]
    for i in range(1, val_length):
        validation_set = pd.concat([validation_set, data[i]], ignore_index=True)
        del data[i]

    test_sets = {}
    for i in range(len(data)):
        test_sets[i] = data[i + val_length]
    del data

    test_prevalences = prevalences.iloc[val_length:, :].copy().reset_index(drop=True)

    return validation_set, test_sets, test_prevalences
