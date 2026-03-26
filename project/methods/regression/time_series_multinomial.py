import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


def softmax(z):
    """
    Computes the softmax of matrix z along the last axis.

    Parameters
    ----------
    z : np.ndarray
        Input array (n_samples, n_classes) containing the logits (i.e., the linear combinations
        of features X and weight matrix C).

    Returns
    -------
    np.ndarray
        Normalized probabilities over the classes for each sample.
        Each row sums to 1, i.e., output shape is (n_samples, n_classes).
    """
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class TimeSeriesMultinomialRegressor:
    """
    A multinomial logistic (softmax) regressor for time series.

    This model maps time information (e.g., days since epoch) to a multinomial probability vector.
    Concretely, time points are first transformed into features (X) using linear, polynomial,
    cyclic basis expansions, or raw time (identity), and min–max normalized using statistics
    computed at fit time as needed. Features are time-only (no constant / bias column).

    Each class k receives a weight vector, assembled in the weight matrix C (shape: d features x K classes).
    For a given sample, the model computes logits by the linear combination `logits = X @ C`.
    Each logit vector ("pre-activation" for each class) is then transformed to a probability
    vector by the softmax function: probability = softmax(logits).

    Fit is performed by minimizing the negative log-likelihood (cross-entropy) between the
    softmax output and the provided soft targets.
    """

    def __init__(self, mode="linear", degree=2, period=None):
        """
        Parameters
        ----------
        mode : str, optional
            Feature construction mode:
            - "linear"      : [t_norm]
            - "polynomial"  : [t_norm, t_norm^2, ..., t_norm^degree]
            - "cyclic"      : [sin(2πt_norm/p), cos(2πt_norm/p)]
            - "identity"    : [t] (no normalization, raw time as feature)
        degree : int, optional
            Degree for the polynomial basis (only used if mode="polynomial").
        period : float or None, optional
            Period for the cyclic basis (used if mode="cyclic").
        """
        self.mode = mode
        self.degree = degree
        self.period = period
        self.C_ = None  # Weight matrix (d_features x K_classes)
        self.t_min_ = None  # Min in training time, for normalization
        self.t_max_ = None  # Max in training time, for normalization

    def _prepare_features(self, t):
        """
        Transform the 1D time vector into a 2D feature matrix X, according
        to the chosen basis expansion and normalization.

        Parameters
        ----------
        t : array-like
            Time values of shape (n_samples,).

        Returns
        -------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        """
        t = np.asanyarray(t).astype(np.float64).flatten()

        if self.mode == "identity":
            return t.reshape(-1, 1)

        # All other modes (linear/poly/cyclic) use normalization
        if self.t_min_ is None or self.t_max_ is None:
            self.t_min_ = float(t.min())
            self.t_max_ = float(t.max())

        range_t = self.t_max_ - self.t_min_
        if range_t > 0:
            t_norm = (t - self.t_min_) / range_t
        else:
            t_norm = np.zeros_like(t)

        t_norm = t_norm.reshape(-1, 1)

        if self.mode == "linear":
            return t_norm

        if self.mode == "polynomial":
            # vander includes leading column of ones; keep only powers of t_norm
            poly = np.vander(t_norm.ravel(), N=self.degree + 1, increasing=True)
            if poly.shape[1] <= 1:
                return t_norm
            return poly[:, 1:]

        if self.mode == "cyclic":
            p = self.period if self.period is not None else 1.0
            if self.period is not None and range_t > 0:
                p = float(self.period) / range_t
            sines = np.sin(2 * np.pi * t_norm / p)
            cosines = np.cos(2 * np.pi * t_norm / p)
            return np.hstack([sines, cosines])

        # Default: just the normalized time
        return t_norm

    def _objective(self, params, X, Y_soft):
        """
        Negative log-likelihood (cross-entropy) using the Log-Softmax trick
        for numerical stability.
        """
        d, K = X.shape[1], Y_soft.shape[1]
        C = params.reshape((d, K))
        
        # 1. Compute the logits (Z = X @ C)
        logits = X @ C  # Shape: (n_samples, n_classes)
        
        # 2. Log-Softmax Trick: log(softmax(z)) = z - log(sum(exp(z)))
        # The logsumexp function stably computes the log of the sum of exponentials.
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        
        # 3. Cross-Entropy: -sum(Y_true * log(P_pred))
        # As log_probs is already the logarithm, we directly multiply by Y_soft.
        # We no longer need log(probs + 1e-15)!
        return -np.sum(Y_soft * log_probs) / X.shape[0]

    def fit(self, t_series, Y_soft):
        """
        Fit the weight matrix C by minimizing the cross-entropy between the target
        soft labels and the predicted multiclass probabilities.

        The input time points t_series are first min–max normalized and expanded into features X,
        except if mode=="identity", in which unix times are used as-is.
        The model learns a weight matrix (C) that linearly combines the features for each class,
        producing logits, which are then passed through the softmax function to obtain final probabilities.

        Parameters
        ----------
        t_series : array-like, shape (n_samples,)
            Time coordinate in days, timestamps, or other float scale.
        Y_soft : np.ndarray, shape (n_samples, n_classes)
            Soft targets on the probability simplex (e.g., classifier probability outputs).

        Returns
        -------
        self : object
            Fitted instance.
        """
        # If using normalization, re-compute min/max when refitting
        if self.mode != "identity":
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
            print(f"Convergence warning (TimeSeriesMultinomialRegressor): {res.message}")

        self.C_ = res.x.reshape((d, K))  # Learned weight matrix.
        return self

    def predict_proba(self, t_series):
        """
        Predict class probabilities for given time points.

        For each time point in t_series, constructs the feature vector,
        computes the linear combination with the weight matrix (logits = X @ C),
        and returns the probability vector obtained by applying the softmax function.

        Parameters
        ----------
        t_series : array-like, shape (n_samples,)
            Time values to predict for.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Predicted probability for each class and sample.
        """
        t_series = np.asanyarray(t_series, dtype=np.float64).flatten()
        X = self._prepare_features(t_series)
        logits = X @ self.C_
        return softmax(logits)

    def predict(self, X):
        """
        Sklearn-style API: Predict class probabilities for a given input array of time values.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1) or (n_samples,)
            Time values to predict for. If 2D, will be converted to 1D.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Probability distributions over classes for each sample.
        """
        X = np.asanyarray(X, dtype=np.float64)
        return self.predict_proba(X.ravel())
