import numpy as np
from sklearn.model_selection import KFold
from Utils import Predictions, Measures

# def online_passive_aggressive_classification(X, y, C):
#     n_samples, n_features = X.shape
#     w = np.zeros(n_features)
#     b = 0
#
#     cost_list = np.array([])
#     epoch_list = np.array([])
#
#     for i in range(n_samples):
#         x = X[i]
#         y_true = y[i]  # y should be +1 or -1 for binary classification
#
#         # Predict the class label (+1 or -1)
#         y_pred = np.dot(w, x) + b
#
#         # Calculate the hinge loss for classification
#         loss = max(0, 1 - y_true * y_pred)
#
#         # Calculate lagrange multiplier T
#         # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
#         # T = min(C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
#         T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3
#
#         # Update weights
#         if loss > 0:
#             w += T * y_true * x
#
#         cost = loss  # cost for classification is simply the hinge loss
#
#         if i % 50 == 0:  # at every 50 iteration record the cost and epoch value
#             cost_list = np.append(cost_list, cost)
#             epoch_list = np.append(epoch_list, i)
#
#     return w, epoch_list, cost_list




def online_passive_aggressive_binary(X, y, C, record_cost_on_every_epoch=50):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0  # Initialize bias term

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    x_minibatch = []
    y_minibatch = np.array([])

    for i in range(1, n_samples):
        x = X[i]
        ys = y[i]
        y_true_hinge = 1 if ys == 1 else -1  # convert to 1 -1 instead of 1 0

        # Calculate prediction
        y_pred = np.dot(w, x) + b

        # Convert to hinge loss
        loss = max(0, 1 - y_true_hinge * y_pred)

        # Calculate lagrange multiplier T for PA3
        T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))

        # Update weights based on the prediction error
        if loss > 0:  # Update only if there's a loss
            w += T * y_true_hinge * x  # Adjust based on the true label
            b += T * y_true_hinge

        x_minibatch.append(x)
        y_minibatch = np.append(y_minibatch, ys)

        if i % record_cost_on_every_epoch == 0:  # Record the cost and epoch value every 50 iterations
            cost_list = np.append(cost_list, loss)
            epoch_list = np.append(epoch_list, i)
            # pa_predicted_y_test = Predictions.predict(x, w, b)  # make sure the order of w is same as coeff
            x_minibatch = np.array(x_minibatch)
            pa_predicted_y_test = Predictions.predict(x_minibatch, w, b)  # make sure the order of w is same as coeff
            pa_acc = Measures.accuracy(y_minibatch, pa_predicted_y_test)
            acc_list = np.append(acc_list, pa_acc)
            x_minibatch = []
            y_minibatch = np.array([])

    return w, b, epoch_list, cost_list, acc_list





def online_passive_aggressive_KFold(X, y, C, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, epoch_list, cost_list, acc_list = online_passive_aggressive_binary(X_train, y_train, C)
        predicted_y_test = Predictions.predict(X_test, w, b)  # make sure the order of w is same as coeff
        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    return acc


def online_passive_aggressive_sk_learn_KFold(X, y, C, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, cost_list, epoch_list  = online_passive_aggressive_sklearn(X_train, y_train, C, seed)
        predicted_y_test = Predictions.predict(X_test, w, b)  # make sure the order of w is same as coeff
        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    return acc








############################################################################


def online_passive_aggressive_sklearn(X, y, C, seed):
    n_samples, n_features = X.shape

    # Initialize the PassiveAggressiveClassifier
    classifier = PassiveAggressiveClassifier(C=C, max_iter=1, tol=1e-3, random_state=seed)

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
        w = classifier.coef_[0]  # Get the updated weights (for binary classification, take the first entry)
        b = classifier.intercept_[0]  # Get the updated bias

        # Optionally compute and store the cost/loss after each update (if needed)
        accuracy = classifier.score(X, y)
        cost_list.append(1 - accuracy)  # Cost can be considered as 1 - accuracy
        epoch_list.append(i + 1)  # Store epoch number (1-based indexing)

    return w, b, cost_list, epoch_list


from sklearn.linear_model import PassiveAggressiveClassifier

def online_passive_aggressive_multiclass_learn_KFold(X, y, C, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, epoch_list, cost_list = online_passive_aggressive_multiclass_sklearn(X_train, y_train, C,seed)
        # predicted_y_test = Predictions.predict(X_test, w, b)  # make sure the order of w is same as coeff
        predicted_y_test = Predictions.predict_multiclass2(X_test, w, b)  # make sure the order of w is same as coeff
        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    return acc

def online_passive_aggressive_multiclass_sklearn(X, y, C, seed):
    n_samples, n_features = X.shape
    classes = np.unique(y)  # Find the unique classes in the labels
    n_classes = len(classes)

    # Initialize the PassiveAggressiveClassifier for multi-class classification
    classifier = PassiveAggressiveClassifier(C=C, max_iter=1, tol=1e-3, random_state=seed)

    # Initialize weight matrix and bias vector
    w = np.zeros((n_classes, n_features))  # Weight matrix for all classes
    b = np.zeros(n_classes)  # Bias vector for all classes

    # Lists to store the cost and epoch information
    cost_list = []
    epoch_list = []

    for i in range(n_samples):
        x = X[i].reshape(1, -1)  # Reshape to match the input shape (1, n_features)
        y_true = y[i]

        # Fit the model incrementally on the current sample
        classifier.partial_fit(x, [y_true], classes=classes)

        # Update the weights and bias after fitting (for all classes)
        w = classifier.coef_  # Get the updated weights (shape: n_classes x n_features)
        b = classifier.intercept_  # Get the updated biases (shape: n_classes)

        # Optionally compute and store the cost/loss after each update (if needed)
        accuracy = classifier.score(X, y)
        cost_list.append(1 - accuracy)  # Cost can be considered as 1 - accuracy
        epoch_list.append(i + 1)  # Store epoch number (1-based indexing)

    return w, b, cost_list, epoch_list


def online_passive_aggressive_multiclass_sklearn2(X, y, C, record_cost_on_every_epoch):
    n_samples, n_features = X.shape
    classes = np.unique(y)  # Find the unique classes in the labels
    n_classes = len(classes)

    # Initialize the PassiveAggressiveClassifier for multi-class classification
    classifier = PassiveAggressiveClassifier(C=C, max_iter=1, tol=1e-3)

    # Initialize weight matrix and bias vector
    w = np.zeros((n_classes, n_features))  # Weight matrix for all classes
    b = np.zeros(n_classes)  # Bias vector for all classes

    # Lists to store the cost and epoch information
    cost_list = []
    epoch_list = []
    acc_list = []
    acc_list2 = []

    x_minibatch = []
    y_minibatch = []
    for i in range(n_samples):
        x = X[i].reshape(1, -1)  # Reshape to match the input shape (1, n_features)
        y_true = y[i]

        x_minibatch.append(X[i])
        y_minibatch.append(y_true)

        # Fit the model incrementally on the current sample
        classifier.partial_fit(x, [y_true], classes=classes)

        # Update the weights and bias after fitting (for all classes)
        w = classifier.coef_  # Get the updated weights (shape: n_classes x n_features)
        b = classifier.intercept_  # Get the updated biases (shape: n_classes)

        if i % record_cost_on_every_epoch == 0:
            # Optionally compute and store the cost/loss after each update (if needed)
            accuracy = classifier.score(np.array(x_minibatch), np.array(y_minibatch))
            # pred_y = Predictions.predict_multiclass2(X,w,b)
            # accuracy2 = Measures.accuracy(y, pred_y) # same as above method
            acc_list = np.append(acc_list, accuracy)
            # acc_list2 = np.append(acc_list2, accuracy2)
            cost_list.append(1 - accuracy)  # Cost can be considered as 1 - accuracy
            epoch_list.append(i + 1)  # Store epoch number (1-based indexing)
            x_minibatch = []
            y_minibatch = []

    return w, b, epoch_list, cost_list, acc_list

