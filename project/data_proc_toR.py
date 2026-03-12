import pandas as pd
import os
import data_loading
from utils import val_test_split
import config


def proc(val, data, name):
    val['TRAININGSET'] = 1
    n_valsubsamples = name[1]

    for i in range(len(data)):
        data[i]['TRAININGSET'] = 0
        comb = pd.concat([val, data[i]], axis=0)
        comb = comb.rename(columns={'label': 'TRUTH', 'text': 'TEXT'})

        valsets_folder = config.README_IMPLEMENT_DIR / 'data' / name[0] / 'val'
        testsets_folder = config.README_IMPLEMENT_DIR / 'data' / name[0] / 'test'
        
        valsets_folder.mkdir(parents=True, exist_ok=True)
        testsets_folder.mkdir(parents=True, exist_ok=True)

        if i < n_valsubsamples:
            comb.to_csv(valsets_folder / f'{i}.csv', index=False)
        else:
            comb.to_csv(testsets_folder / f'{i-n_valsubsamples}.csv', index=False)
    return


def proc_single(val, test):
    val['TRAININGSET'] = 1
    test['TRAININGSET'] = 0
    comb = pd.concat([val, test], axis=0).rename(columns={'label': 'TRUTH', 'text': 'TEXT'})
    return comb


text_senti_data = [('global_covid19_tweets', 15), ('nepali_dataset_eng', 15), ('Apple-Twitter-Sentiment-DFE', 15)]

for data_name in text_senti_data:
    ts_chunks, ts_prevalence, c, t_size = data_loading.loading(data_name[0])
    val_set, test_sets, test_dsts = val_test_split(ts_chunks.copy(), ts_prevalence, data_name[1])
    proc(val_set, ts_chunks, data_name)
