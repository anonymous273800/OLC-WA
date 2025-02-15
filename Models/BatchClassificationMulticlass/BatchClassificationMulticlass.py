from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Utils import Predictions, Measures

def logistic_regression_multiclass(xs, ys):
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    model.fit(xs, ys)
    w = model.coef_  # Now w is a matrix with shape (n_classes, n_features)
    b = model.intercept_  # Now b is an array with shape (n_classes,)
    return w, b


def batch_logistic_regression_multiclass_KFold(X, y, seed):
    # kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b = logistic_regression_multiclass(X_train, y_train)

        # Update prediction logic for multiclass
        predicted_y_test = Predictions.predict_multiclass(X_test, w, b)

        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    return acc