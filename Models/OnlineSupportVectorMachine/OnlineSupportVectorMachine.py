import warnings
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from Utils import Measures, Predictions
warnings.filterwarnings("ignore")


def online_support_vector_machine_sklearn_KFold(X, y, C, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, cost_list, epoch_list = online_support_vector_machine_sklearn(X_train, y_train, C, seed)
        predicted_y_test = Predictions.predict(X_test, w, b)  # make sure the order of w is same as coeff
        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    return acc

def online_support_vector_machine_sklearn(X, y, C, seed):
    n_samples, n_features = X.shape

    # Initialize the SGDClassifier with hinge loss (SVM)
    classifier = SGDClassifier(loss='hinge', alpha=1/C, max_iter=1, tol=None, random_state=seed)

    # Initialize weight vector and bias
    w = np.zeros(n_features)  # Weight vector for the features
    b = 0  # Bias term

    # Lists to store the cost and epoch information
    cost_list = []
    epoch_list = []

    for i in range(n_samples):
        x = X[i].reshape(1, -1)  # Reshape to match the input shape (1, n_features)
        y_true = y[i]

        # Fit the model incrementally on the current sample
        classifier.partial_fit(x, [y_true], classes=np.unique(y))

        # Update the weights and bias after fitting
        w = classifier.coef_[0]  # Get the updated weights
        b = classifier.intercept_[0]  # Get the updated bias

        # Optionally compute and store the cost/loss after each update
        accuracy = classifier.score(X, y)
        cost_list.append(1 - accuracy)  # Cost can be considered as 1 - accuracy
        epoch_list.append(i + 1)  # Store epoch number (1-based indexing)

    return w, b, cost_list, epoch_list