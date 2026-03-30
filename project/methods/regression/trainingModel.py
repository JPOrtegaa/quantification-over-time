import numpy as np
from sklearn.linear_model import LinearRegression

import config
from .time_series_multinomial import TimeSeriesMultinomialRegressor


def trainer(train_L, train_predictions, model_name, seed, **trainer_kw):
    if model_name == "LR":
        regressor = LinearRegression(n_jobs=getattr(config, "SKLEARN_N_JOBS", 1))
        regressor.fit(train_L, train_predictions)
        return regressor
    
    if model_name == "TSMN":
        mode = trainer_kw.get("tsmn_mode", "polynomial")
        degree = trainer_kw.get("tsmn_degree", 3)
        period = trainer_kw.get("tsmn_period")
        regressor = TimeSeriesMultinomialRegressor(mode=mode, degree=degree, period=period)
        train_L = np.asanyarray(train_L, dtype=np.float64)
        # (n, d) with d>1: fixed design matrix (e.g. weekday one-hot); do not ravel
        if train_L.ndim == 2 and train_L.shape[1] > 1:
            t = train_L
        elif train_L.ndim == 2 and train_L.shape[1] == 1:
            t = train_L.ravel()
        else:
            t = train_L.ravel()
        regressor.fit(t, train_predictions)
        return regressor

    raise ValueError(f"Unknown regressor model_name: {model_name!r}")
