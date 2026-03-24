
# regressors
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC


def trainer(train_L, train_predictions, model_name, seed):

    regressor = None
    if model_name == "LR":
        regressor = LinearRegression(n_jobs=-1)

    regressor.fit(train_L, train_predictions)

    return regressor
