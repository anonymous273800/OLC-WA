import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import Utils.Util
from Utils import Measures, Predictions
from itertools import combinations


def perceptron(X_train, y_train, record_cost_on_every_epoch=50):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    x_minibatch = []
    y_minibatch = []

    cost = 0
    y_train = np.where(y_train==0, -1, 1)


    for i in range(1, n_samples):
        xs = X_train[i]
        ys = y_train[i]

        x_minibatch.append(xs)
        y_minibatch.append(ys)

        linear_model = np.dot(xs, w) + b
        from scipy.sparse import issparse


        # Apply the sign function to the dense array
        y_predicted = np.sign(linear_model)



        if (ys != y_predicted):
            w = w +  ys*xs
            b = b +  ys
            cost+=1

        # Record cumulative cost and epoch values at intervals (every 50 iterations)
        if i % record_cost_on_every_epoch == 0:
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

            x_minibatch = np.array(x_minibatch)
            y_minibatch = np.array(y_minibatch)
            # Calculate predictions for the minibatch
            y_minibatch_predicted = np.where(np.dot(x_minibatch, w) + b >= 0, 1, -1)
            acc = Measures.accuracy(y_minibatch, y_minibatch_predicted)
            acc_list = np.append(acc_list, acc)

            # Reset minibatch for the next iteration
            x_minibatch = []
            y_minibatch = []

    return w, b, epoch_list, cost_list, acc_list


def perceptron_KFold(X, y, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w,b,epoch_list, cost_list, acc_list = perceptron(X_train, y_train)
        y_predicted = Predictions.predict_widrow_hoff(X_test, w, b)  # TODO: check why not predict_perceptron here? this looks to me a problem
        acc_per_split_for_same_seed = Measures.accuracy(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    return acc

#######################################OVR#############################################

def perceptron_ovr_classification(X_train, y_train):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    cost = 0
    cost_list = np.array([])
    epoch_list = np.array([])
    for i in range(n_samples):
        xs = X_train[i]
        ys = y_train[i]
        linear_model = np.dot(xs, w) + b
        y_predicted = 1 if linear_model >= 0 else -1 #np.sign(linear_model)
        if ys != y_predicted:
            w = w + ys * xs
            b = b + ys
            cost+=1

        # Record cumulative cost and epoch values at intervals (every 50 iterations)
        if i % 50 == 0:
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, b, epoch_list, cost_list

def perceptron_ovr(X_train, y_train):
    n_classes = len(np.unique(y_train))
    classifiers = []
    all_epoch_lists = []
    all_cost_lists = []
    for class_idx in range(n_classes):
        # Create binary labels for one-vs-rest classification
        y_binary = np.where(y_train == class_idx, 1, -1)
        w, b, epoch_list, cost_list = perceptron_ovr_classification(X_train, y_binary)
        classifiers.append((w,b))
        all_epoch_lists.append(epoch_list)
        all_cost_lists.append(cost_list)
    return classifiers, all_epoch_lists, all_cost_lists



def perceptron_ovr_KFold(X, y, seed):
    accuracy_list = []  # To store accuracy scores for each fold
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifiers, epoch_list, costs_list = perceptron_ovr(X_train, y_train)

        # Predict using the one-vs-rest classifiers
        y_pred = []
        for i in range(len(X_test)):
            scores = np.array([np.dot(X_test[i], w) + b for w, b in classifiers])
            predicted_class = np.argmax(scores)  # Choose the class with the highest score
            y_pred.append(predicted_class)

        # Calculate accuracy
        accuracy = Measures.accuracy(y_test, y_pred)
        accuracy_list.append(accuracy)

    # Return the average accuracy across all folds
    acc = np.mean(accuracy_list)
    return acc

#########################################OVO#############################################
def perceptron_ovo_classification(X_train, y_train, class_1, class_2):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    cost = 0
    cost_list = np.array([])  # To store the cost at each iteration
    epoch_list = np.array([])  # To store the epoch (iteration) value at intervals

    # Create binary labels for the two classes (1 if class_1, -1 if class_2)
    y_binary = np.where(y_train == class_1, 1, -1)

    for i in range(n_samples):
        xs = X_train[i]
        ys = y_binary[i]
        linear_model = np.dot(xs, w) + b
        y_predicted  = 1 if linear_model >= 0 else -1 #np.sign(linear_model)

        if ys != y_predicted:
            w = w + ys * xs
            b = b + ys
            cost += 1

        # Record cumulative cost and epoch values at intervals (every 50 iterations)
        if i % 50 == 0:
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, b, epoch_list, cost_list


def perceptron_ovo(X_train, y_train):
    n_classes = len(np.unique(y_train))
    classifiers = []
    all_epoch_lists = []
    all_cost_lists = []

    # Generate all pairs of classes using itertools.combinations
    class_pairs = list(combinations(range(n_classes), 2))

    for class_1, class_2 in class_pairs:
        w, b, epoch_list, cost_list = perceptron_ovo_classification(X_train, y_train, class_1, class_2)
        classifiers.append((w, b, class_1, class_2))
        all_epoch_lists.append(epoch_list)
        all_cost_lists.append(cost_list)

    return classifiers, all_epoch_lists, all_cost_lists


def perceptron_ovo_KFold(X, y, seed):
    accuracy_list = []  # To store accuracy scores for each fold
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    n_classes = len(np.unique(y))

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        classifiers, epoch_list, costs_list = perceptron_ovo(X_train, y_train)

        # Predict using the one-vs-one classifiers (voting scheme)
        y_pred = []
        for i in range(len(X_test)):
            # Votes for each class
            votes = np.zeros(n_classes)  # Initialize a vote counter for each class
            for w, b, class_1, class_2 in classifiers:
                linear_model = np.dot(X_test[i], w) + b
                y_predicted  = 1 if linear_model >= 0 else -1  # np.sign
                predicted_class = class_1 if y_predicted == 1 else class_2
                votes[predicted_class] += 1  # Vote for the predicted class

            # The class with the highest vote wins
            predicted_class = np.argmax(votes)
            y_pred.append(predicted_class)

        # Calculate accuracy
        accuracy = Measures.accuracy(y_test, y_pred)
        accuracy_list.append(accuracy)

    # Return the average accuracy across all folds
    acc = np.mean(accuracy_list)
    return acc


################################################################################################

# def predict_perceptron_mc(classifiers, x_minibatch, y_minibatch, n_classes):
#     y_pred_batch = []
#     for xs in x_minibatch:
#         scores = []
#         for class_inx in range(n_classes):
#             w, b = classifiers[class_inx]
#
#             # Calculate the linear score for the current sample and class
#             score = w.dot(xs) + b
#
#             # Append this score to the scores list for the current sample
#             scores.append(score)
#
#         # Step 2: Determine the predicted class for the current sample
#         # Take the index of the highest score as the predicted class
#         predicted_class = np.argmax(scores)
#
#         # Step 3: Store the predicted class in the y_pred_batch list
#         y_pred_batch.append(predicted_class)
#
#     # Step 4: Calculate accuracy for the minibatch
#     # Convert y_minibatch and y_pred_batch to arrays for compatibility with the accuracy function
#     acc = Measures.accuracy(np.array(y_minibatch), np.array(y_pred_batch))
#     return acc

def perceptron_mc_ovr(X_train, y_train, record_cost_at_each_epochs):
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    epoch_list = []
    cost_list = []
    acc_list = []
    classifiers = [(np.zeros(n_features), 0) for _ in range(n_classes)]  # Initialize weights and biases
    x_minibatch = []
    y_minibatch = []
    cost = 0

    for i in range(n_samples):
        xs = X_train[i]
        true_class = y_train[i]
        x_minibatch.append(xs)
        y_minibatch.append(true_class)


        # Iterate over each class for OvR
        for class_idx in range(n_classes):
            w, b = classifiers[class_idx]
            ys = 1 if true_class == class_idx else -1  # Binary target for OvR
            linear_model = np.dot(xs, w) + b
            y_predicted = 1 if linear_model >= 0 else -1

            # Update weights if there is a misclassification
            if ys != y_predicted:
                w = w + ys * xs
                b = b + ys
                cost += 1
                classifiers[class_idx] = (w, b)  # Update the classifier


        # Record metrics at specified intervals
        if i % record_cost_at_each_epochs == 0:
            cost_list.append(cost)
            epoch_list.append(i + 1)

            # Calculate accuracy on the minibatch
            acc = Measures.perceptron_acc_mc_ovr(classifiers, x_minibatch, y_minibatch)
            # y_pred_minibatch = Measures.predict_perceptron_mc(classifiers, x_minibatch, y_minibatch, n_classes)

            # acc = Measures.accuracy(np.array(y_minibatch), np.array(y_pred_minibatch))
            acc_list.append(acc)

            # Clear minibatch lists
            x_minibatch = []
            y_minibatch = []

    return classifiers, epoch_list, cost_list, acc_list

# def perceptron_mc_ovr(X_train, y_train, record_cost_at_each_epochs):
#     n_samples, n_features = X_train.shape
#     n_classes = len(np.unique(y_train))
#     epoch_list = []
#     cost_list = []
#     acc_list = []
#     classifiers = [(np.zeros(n_features), 0) for _ in range(n_classes)]  # Initialize weights and biases
#     x_minibatch = []
#     y_minibatch = []
#     y_pred_minibatch = []
#     cost = 0
#
#     for i in range(n_samples):
#         xs = X_train[i]
#         true_class = y_train[i]
#         x_minibatch.append(xs)
#         y_minibatch.append(true_class)
#
#         # Track the prediction for this sample across all classifiers
#         sample_predictions = []
#
#         # Iterate over each class for OvR
#         for class_idx in range(n_classes):
#             w, b = classifiers[class_idx]
#             ys = 1 if true_class == class_idx else -1  # Binary target for OvR
#             linear_model = np.dot(xs, w) + b
#             # Store the prediction score for the current class
#             sample_predictions.append(linear_model)
#
#             y_predicted = 1 if linear_model >= 0 else -1
#
#             # Update weights if there is a misclassification
#             if ys != y_predicted:
#                 w = w + ys * xs
#                 b = b + ys
#                 cost += 1
#                 classifiers[class_idx] = (w, b)  # Update the classifier
#
#         # Append the predicted class (class with the highest score) for accuracy calculation
#         predicted_class = np.argmax(sample_predictions)
#         y_pred_minibatch.append(predicted_class)
#
#         # Record metrics at specified intervals
#         if i % record_cost_at_each_epochs == 0:
#             cost_list.append(cost)
#             epoch_list.append(i + 1)
#
#             # Calculate accuracy on the minibatch
#             acc = Measures.accuracy(np.array(y_minibatch), np.array(y_pred_minibatch))
#             acc_list.append(acc)
#
#             # Clear minibatch lists
#             x_minibatch = []
#             y_minibatch = []
#             y_pred_minibatch = []
#
#     return classifiers, epoch_list, cost_list, acc_list
