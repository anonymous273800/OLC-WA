import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from river import tree
from river import metrics
from sklearn.model_selection import KFold
from Utils import Measures, Predictions

# def hoeffding_tree(X_train, y_train, record_cost_on_every_epoch=10):
#     n_samples, n_features = X_train.shape
#
#     cost_list = np.array([])
#     epoch_list = np.array([])
#     acc_list = np.array([])
#
#     x_minibatch = []
#     y_minibatch = []
#
#     # Create the Hoeffding Tree classifier
#     model = tree.HoeffdingTreeClassifier()
#
#     # Train the model using the training data
#     for i in range(n_samples):
#         # Create a dictionary for the current instance
#
#         x_instance = {f'feature_{j}': X_train[i, j] for j in range(n_features)}
#         x_minibatch.append(x_instance)
#         y_minibatch.append(y_train[i])
#
#         # Update the model with the current instance
#         model.learn_one(x_instance, y_train[i])
#
#         if i % record_cost_on_every_epoch == 0:
#             pass
#         # use x_minibatch and y_minibatch to compute loss, accuracy
#
#
#
#     return model, epoch_list, cost_list, acc_list

import numpy as np
from river import tree
from river import metrics

# def hoeffding_tree(X_train, y_train, record_cost_on_every_epoch=10):
#     n_samples, n_features = X_train.shape
#
#     cost_list = np.array([])
#     epoch_list = np.array([])
#     acc_list = np.array([])
#
#     x_minibatch = []
#     y_minibatch = []
#
#     # Create the Hoeffding Tree classifier
#     model = tree.HoeffdingTreeClassifier()
#
#     # Initialize accuracy and log-loss metrics
#     accuracy_metric = metrics.Accuracy()
#     log_loss_metric = metrics.LogLoss()
#
#     # Train the model using the training data
#     for i in range(n_samples):
#         # Create a dictionary for the current instance
#         x_instance = {f'feature_{j}': X_train[i, j] for j in range(n_features)}
#         x_minibatch.append(x_instance)
#         y_minibatch.append(y_train[i])
#
#         # Update the model with the current instance
#         model.learn_one(x_instance, y_train[i])
#
#         # Make a prediction and update metrics
#         y_pred = model.predict_one(x_instance)
#         accuracy_metric.update(y_train[i], y_pred)
#         y_pred_proba = model.predict_proba_one(x_instance)
#         log_loss_metric.update(y_train[i], y_pred_proba)
#
#         # Record cost and accuracy every `record_cost_on_every_epoch` iterations
#         if i % record_cost_on_every_epoch == 0:
#             epoch_list = np.append(epoch_list, i)
#
#             # Calculate the mean log-loss for the current minibatch
#             mean_loss = log_loss_metric.get()
#             cost_list = np.append(cost_list, mean_loss)
#
#             # Calculate accuracy for the current minibatch
#             accuracy_value = accuracy_metric.get()
#             acc_list = np.append(acc_list, accuracy_value)
#
#     return model, epoch_list, cost_list, acc_list


from river import tree
from river import metrics


def hoeffding_tree(X_train, y_train, record_cost_on_every_epoch=10):
    n_samples, n_features = X_train.shape

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    x_minibatch = []
    y_minibatch = []

    # Create the Hoeffding Tree classifier
    model = tree.HoeffdingTreeClassifier()

    # Train the model using the training data
    for i in range(1, n_samples):
        # Create a dictionary for the current instance
        x_instance = {f'feature_{j}': X_train[i, j] for j in range(n_features)}
        x_minibatch.append(x_instance)
        y_minibatch.append(y_train[i])

        # Update the model with the current instance
        model.learn_one(x_instance, y_train[i])

        if i % record_cost_on_every_epoch == 0:
            epoch_list = np.append(epoch_list, (i + 1))

            # Calculate accuracy for the current minibatch
            y_predictions = [model.predict_one(x) for x in x_minibatch]
            accuracy = Measures.accuracy(np.array(y_minibatch), np.array(y_predictions))
            acc_list = np.append(acc_list, accuracy)

            # Calculate the mean log-loss for the current minibatch
            log_losses = []
            for x, y_true in zip(x_minibatch, y_minibatch):
                y_pred_proba = model.predict_proba_one(x)
                if y_true in y_pred_proba:
                    log_losses.append(-np.log(y_pred_proba[y_true]))
                else:
                    log_losses.append(np.inf)  # Handle cases where the class is not predicted
            mean_loss = np.mean(log_losses)
            cost_list = np.append(cost_list, mean_loss)

            # Clear minibatch after recording the metrics
            x_minibatch = []
            y_minibatch = []

    return model, epoch_list.astype(int), cost_list, acc_list


def hoeffding_tree_KFold(X, y, seed):
    n_samples, n_features = X.shape
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model, epoch_list, cost_list, acc_list = hoeffding_tree(X_train, y_train)

        # y_predicted = []
        # for i in range(len(X_test)):
        #     x_test_instance = {f'feature_{j}': X_test[i, j] for j in range(n_features)}
        #     y_test_pred = model.predict_one(x_test_instance)
        #     y_predicted.append(y_test_pred)
        y_predicted = Predictions.predict_hoefding_tree(X_test, model)
        acc_per_split_for_same_seed = Measures.accuracy(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    return acc