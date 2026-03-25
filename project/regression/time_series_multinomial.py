import numpy as np
from scipy.optimize import minimize


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class TimeSeriesMultinomialRegressor:
    """
    Regressor that maps Unix timestamps (seconds) to a probability simplex via
    logits = X @ C and probs = softmax(logits). Features X are built from time
    with optional linear / polynomial / cyclic bases; time is min–max normalized
    using statistics stored at fit time.
    """

    def __init__(self, mode="linear", degree=2, period=None):
        self.mode = mode
        self.degree = degree
        self.period = period
        self.C_ = None
        self.t_min_ = None
        self.t_max_ = None

    def _prepare_features(self, t):
        t = np.asanyarray(t).astype(np.float64).flatten()

        if self.t_min_ is None or self.t_max_ is None:
            self.t_min_ = float(t.min())
            self.t_max_ = float(t.max())

        range_t = self.t_max_ - self.t_min_
        if range_t > 0:
            t_norm = (t - self.t_min_) / range_t
        else:
            t_norm = np.zeros_like(t)

        t_norm = t_norm.reshape(-1, 1)
        ones = np.ones_like(t_norm)

        if self.mode == "linear":
            return np.hstack([ones, t_norm])

        if self.mode == "polynomial":
            return np.vander(t_norm.ravel(), N=self.degree + 1, increasing=True)

        if self.mode == "cyclic":
            p = self.period if self.period is not None else 1.0
            if self.period is not None and range_t > 0:
                p = float(self.period) / range_t
            sines = np.sin(2 * np.pi * t_norm / p)
            cosines = np.cos(2 * np.pi * t_norm / p)
            return np.hstack([ones, sines, cosines])

        return t_norm

    def _objective(self, params, X, Y_soft):
        d, K = X.shape[1], Y_soft.shape[1]
        C = params.reshape((d, K))
        logits = X @ C
        probs = softmax(logits)
        return -np.sum(Y_soft * np.log(probs + 1e-15)) / X.shape[0]

    def fit(self, t_series, Y_soft):
        """
        t_series: timestamps in seconds (1d or column vector)
        Y_soft: (n_samples, n_classes) soft targets on the simplex (e.g. classifier probs)
        """
        self.t_min_ = None
        self.t_max_ = None
        X = self._prepare_features(t_series)
        d, K = X.shape[1], Y_soft.shape[1]

        initial_guess = np.zeros(d * K)
        res = minimize(
            self._objective,
            initial_guess,
            args=(X, np.asanyarray(Y_soft, dtype=np.float64)),
            method="L-BFGS-B",
            options={"disp": False},
        )
        if not res.success:
            print(f"Aviso de convergência (TimeSeriesMultinomialRegressor): {res.message}")

        self.C_ = res.x.reshape((d, K))
        return self

    def predict_proba(self, t_series):
        t_series = np.asanyarray(t_series, dtype=np.float64).flatten()
        X = self._prepare_features(t_series)
        logits = X @ self.C_
        return softmax(logits)

    def predict(self, X):
        """Sklearn-style API: X is (n, 1) Unix seconds, as from _time_column_to_X."""
        X = np.asanyarray(X, dtype=np.float64)
        return self.predict_proba(X.ravel())
