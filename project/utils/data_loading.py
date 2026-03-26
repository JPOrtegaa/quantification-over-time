import pandas as pd
import numpy as np
import chardet
from io import BytesIO
from zipfile import ZipFile
import urllib.request
from utils import val_test_split
import config

WINDOW_SIZE_RATIO = 0.1  # 10% of dataset
STRIDE_RATIO = 0.01      # 1% of dataset

def sliding_window_split(df, classes):
    dataset_size = len(df)
    window_size = max(10, int(dataset_size * WINDOW_SIZE_RATIO))
    stride = max(1, int(dataset_size * STRIDE_RATIO))
    
    # Calculate stride_ratio relative to window_size for temporal constraint
    # stride_ratio = (stride / window_size) represents the fraction of "new" data per window
    stride_ratio = stride / window_size

    ts_chunks = {}
    chunk_index = 0
    start = 0
    
    prevalences = {c: [] for c in classes}
    
    while start + window_size <= len(df):
        window_df = df.iloc[start : start + window_size].copy()
        ts_chunks[chunk_index] = window_df
        
        # Calculate prevalence
        label_counts = window_df["label"].value_counts()
        for c in classes:
            count = label_counts.get(c, 0)
            prevalences[c].append(count / window_size)
            
        start += stride
        chunk_index += 1
        
    prevalence_df = pd.DataFrame(prevalences)
    
    # ensure columns match classes order exactly
    prevalence_df = prevalence_df[classes]
    
    return ts_chunks, prevalence_df, stride_ratio


def count_median(datadict):
    count_median = []
    for i in datadict:
        count_median.append(len(datadict[i]))
    if len(count_median) == 0:
        return 0
    median_size = np.median(np.array(count_median))
    return int(median_size)


def nepali_dataset_eng():
    df1 = pd.read_csv(config.DATA_DIR / "Nepali_dataset_Eng.csv")
    df1 = df1.drop(labels=["Unnamed: 0", "Tweet", "Tokanize_tweet"], axis=1)
    neworder = ["Label", "Tweet_en", "Datetime"]
    df1 = df1.reindex(columns=neworder)
    df1 = df1.rename(columns={"Label": "label", "Tweet_en": "text"})
    df1 = df1[df1["label"].isin([-1, 0, 1])]
    
    # Sort chronologically
    df1 = df1.sort_values(by="Datetime").reset_index(drop=True)

    classes = [-1, 0, 1]
    ts_chunks, prevalence_df, stride_ratio = sliding_window_split(df1, classes)

    return ts_chunks, prevalence_df, classes, stride_ratio


def global_covid19_tweets():
    training_set = (
        config.DATA_DIR
        / "global_covid19_tweets"
        / "global_covid19_tweets"
        / "Corona_NLP_train.csv"
    )
    test_set = (
        config.DATA_DIR
        / "global_covid19_tweets"
        / "global_covid19_tweets"
        / "Corona_NLP_test.csv"
    )
    df = pd.read_csv(training_set)
    df_test = pd.read_csv(test_set)
    df1 = pd.concat([df_test, df])
    df1 = df1.drop(labels=["UserName", "ScreenName", "Location"], axis=1)
    neworder = ["Sentiment", "OriginalTweet", "TweetAt"]
    df1 = df1.reindex(columns=neworder)
    df1.loc[df1["Sentiment"] == "Extremely Positive", "Sentiment"] = 1
    df1.loc[df1["Sentiment"] == "Extremely Negative", "Sentiment"] = -1
    df1.loc[df1["Sentiment"] == "Positive", "Sentiment"] = 1
    df1.loc[df1["Sentiment"] == "Negative", "Sentiment"] = -1
    df1.loc[df1["Sentiment"] == "Neutral", "Sentiment"] = 0
    df1 = df1[df1["Sentiment"].isin([0, 1, -1])]
    df1 = df1.rename(columns={"Sentiment": "label", "OriginalTweet": "text"})
    
    # Format date and sort chronologically
    df1["TweetAt"] = pd.to_datetime(df1["TweetAt"], format="%d-%m-%Y")
    df1 = df1.sort_values(by="TweetAt").reset_index(drop=True)

    classes = [-1, 0, 1]
    ts_chunks, prevalence_df, stride_ratio = sliding_window_split(df1, classes)

    return ts_chunks, prevalence_df, classes, stride_ratio


def Apple_Twitter_Sentiment_DFE():
    data_path = config.DATA_DIR / "Apple-Twitter-Sentiment-DFE.csv"
    with open(data_path, "rb") as f:
        enc = chardet.detect(f.read())
    df = pd.read_csv(data_path, encoding=enc["encoding"])
    df1 = df.drop(
        labels=[
            "_unit_id",
            "_golden",
            "_unit_state",
            "_trusted_judgments",
            "_last_judgment_at",
            "sentiment:confidence",
            "id",
            "query",
            "sentiment_gold",
        ],
        axis=1,
    )
    df1 = df1[df1["sentiment"].isin(["1", "3", "5"])]

    df1["date"] = df1["date"].str[:13]
    df1["date"] = (
        df1["date"].str.split(" ").str[2].astype(int) * 24
        + df1["date"].str.split(" ").str[3].astype(int)
        - 43
    ) // 6
    neworder = ["sentiment", "text", "date"]
    df1 = df1.reindex(columns=neworder)
    df1.loc[df1["sentiment"] == "5", "sentiment"] = 1
    df1.loc[df1["sentiment"] == "3", "sentiment"] = -1
    df1.loc[df1["sentiment"] == "1", "sentiment"] = 0
    df1 = df1.rename(columns={"sentiment": "label"})

    # Sort chronologically by the custom integer date
    df1 = df1.sort_values(by="date").reset_index(drop=True)

    classes = [-1, 0, 1]
    ts_chunks, prevalence_df, stride_ratio = sliding_window_split(df1, classes)

    return ts_chunks, prevalence_df, classes, stride_ratio


def bike():
    training_size = 38
    f = config.DATA_DIR / "bike_sharing_dataset" / "hour.csv"
    dta = pd.read_csv(f, header=0, skipinitialspace=True)
    dta = dta.drop(["instant", "casual", "registered"], axis=1)
    dta = pd.get_dummies(
        dta, columns=["season", "yr", "mnth", "hr", "weekday", "weathersit"]
    )

    bins = [0, 100, 1000]
    labels = [0, 1]
    dta["cnt"] = pd.cut(dta["cnt"], bins=bins, labels=labels)
    dta["cnt"] = dta["cnt"].astype("int64")
    dta = dta.rename(columns={"cnt": "label"})

    # dataset is already chronologically ordered in hour.csv, but we sort by dteday just in case 
    dta["dteday"] = pd.to_datetime(dta["dteday"])
    dta = dta.sort_values(by="dteday").reset_index(drop=True)

    # We need to drop dteday since it can't be used as a numerical feature directly
    dta = dta.drop(["dteday"], axis=1)

    classes = [0, 1]
    ts_chunks, prevalence_df, stride_ratio = sliding_window_split(dta, classes)

    training_data, ts_data_dict, ts_prevalence = val_test_split(
        ts_chunks.copy(), prevalence_df, training_size
    )
    training_data = training_data.sample(
        frac=1.0, replace=False, random_state=42
    ).reset_index(drop=True)

    return training_data, ts_data_dict, ts_prevalence, classes, stride_ratio


def energy():
    training_size = 15
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"

    dta = pd.read_csv(url, header=0, skipinitialspace=True)
    dta = dta.drop(["rv1", "rv2"], axis=1)

    bins = [0, 50, 100, 2000]
    labels = [-1, 0, 1]
    dta["Appliances"] = pd.cut(dta["Appliances"], bins=bins, labels=labels)
    dta["Appliances"] = dta["Appliances"].astype("int64")
    dta = dta.rename(columns={"Appliances": "label"})

    # dta is chronologically ordered, sort by date
    dta["date"] = pd.to_datetime(dta["date"])
    dta = dta.sort_values(by="date").reset_index(drop=True)
    dta = dta.drop(["date"], axis=1)

    classes = [-1, 0, 1]
    ts_chunks, prevalence_df, stride_ratio = sliding_window_split(dta, classes)

    training_data, ts_data_dict, ts_prevalence = val_test_split(
        ts_chunks.copy(), prevalence_df, training_size
    )
    training_data = training_data.sample(
        frac=1.0, replace=False, random_state=42
    ).reset_index(drop=True)

    return training_data, ts_data_dict, ts_prevalence, classes, stride_ratio


def news():
    training_size = 21
    url = urllib.request.urlopen(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip"
    )

    my_zip_file = ZipFile(BytesIO(url.read()))
    dta_file = my_zip_file.namelist()[2]

    dta = pd.read_csv(my_zip_file.open(dta_file), skipinitialspace=True)
    dta = dta.drop(columns="url")

    bins = [0, 1000, 1000000]
    labels = [0, 1]
    dta["shares"] = pd.cut(dta["shares"], bins=bins, labels=labels)
    dta["shares"] = dta["shares"].astype("int64")
    dta = dta.rename(columns={"shares": "label"})

    # Sort by timedelta (reverse chronological in original: "Days between the article publication and the dataset acquisition")
    # To get chronological order, we sort descending by timedelta
    dta = dta.sort_values(by="timedelta", ascending=False).reset_index(drop=True)
    dta = dta.drop(["timedelta"], axis=1)

    classes = [0, 1]
    ts_chunks, prevalence_df, stride_ratio = sliding_window_split(dta, classes)

    training_data, ts_data_dict, ts_prevalence = val_test_split(
        ts_chunks.copy(), prevalence_df, training_size
    )
    training_data = training_data.sample(
        frac=1.0, replace=False, random_state=42
    ).reset_index(drop=True)

    return training_data, ts_data_dict, ts_prevalence, classes, stride_ratio


def loading(dataname):

    if dataname == "nepali_dataset_eng":
        return nepali_dataset_eng()

    elif dataname == "global_covid19_tweets":
        return global_covid19_tweets()

    elif dataname == "Apple-Twitter-Sentiment-DFE":
        return Apple_Twitter_Sentiment_DFE()

    elif dataname == "bike":
        return bike()

    elif dataname == "energy":
        return energy()

    elif dataname == "news":
        return news()
