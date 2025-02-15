import numpy as np
from sklearn.model_selection import KFold
from Utils import Measures, Predictions, Util, Constants
import itertools


def widrow_hoff_classification(X, y, learning_rate, record_cost_on_every_epoch=50):
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights without bias
    b = 0  # Initialize bias term

    # Convert labels to {-1, 1} for binary classification
    y = np.where(y == 0, -1, 1)


    x_minibatch = []
    y_minibatch = []

    for i in range(1, n_samples):
        x_i = X[i]
        y_i = y[i]

        x_minibatch.append(x_i)
        y_minibatch.append(y_i)

        # Calculate the prediction (dot product of w and x_i plus bias)
        y_pred = np.dot(w, x_i) + b

        # Update rule: w <- w + learning_rate * (y_i - y_pred) * x_i
        w = w + learning_rate * (y_i - y_pred) * x_i
        # Update bias: b <- b + learning_rate * (y_i - y_pred)
        b = b + learning_rate * (y_i - y_pred)

        # # Calculate cost (squared error of raw output)
        # cost = np.square(y_i - y_pred)

        if i % record_cost_on_every_epoch == 0:  # Record cost and epoch value every 50 iterations
            epoch_list = np.append(epoch_list, i)

            x_minibatch = np.array(x_minibatch)
            y_minibatch = np.array(y_minibatch)

            y_pred = np.dot(x_minibatch, w ) + b
            y_pred_labels = np.where(y_pred >= 0, 1, -1)
            mse = np.mean((y_minibatch - y_pred_labels) ** 2)
            cost_list = np.append(cost_list, mse)

            acc = Measures.accuracy(y_minibatch,y_pred_labels)
            acc_list = np.append(acc_list, acc)

            x_minibatch = []
            y_minibatch = []

    return w, b, epoch_list, cost_list, acc_list  # Return weights and bias


def widrow_hoff_classification_ovr_ovo(X, y, learning_rate):
    cost_list = np.array([])
    epoch_list = np.array([])

    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # Initialize weights without bias
    b = 0  # Initialize bias term



    for i in range(n_samples):
        x_i = X[i]
        y_i = y[i]

        # Calculate the prediction (dot product of w and x_i plus bias)
        y_pred = np.dot(w, x_i) + b

        # Update rule: w <- w + learning_rate * (y_i - y_pred) * x_i
        w = w + learning_rate * (y_i - y_pred) * x_i
        # Update bias: b <- b + learning_rate * (y_i - y_pred)
        b = b + learning_rate * (y_i - y_pred)

        # Calculate cost (squared error of raw output)
        cost = np.square(y_i - y_pred)

        if i % 50 == 0:  # Record cost and epoch value every 50 iterations
            cost_list = np.append(cost_list, cost)
            epoch_list = np.append(epoch_list, i)

    return w, b, epoch_list, cost_list  # Return weights and bias


def widrow_hoff_KFold(X, y, learning_rate, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, epoch_list, cost_list, acc_list = widrow_hoff_classification(X_train, y_train, learning_rate)
        # y_predicted = Predictions.predict(X_test, w, b)
        y_predicted = Predictions.predict_widrow_hoff(X_test, w, b)
        acc_per_split_for_same_seed = Measures.accuracy(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    return acc


def widrow_hoff_OvR(X_train, y_train, learning_rate):
    n_classes = len(np.unique(y_train))
    classifiers = []
    all_epoch_lists = []
    all_cost_lists = []
    for class_idx in range(n_classes):
        # Create binary labels for one-vs-rest classification
        y_binary = np.where(y_train == class_idx, 1, -1)

        # Train Widrow-Hoff classifier for this class
        w, b, epoch_list, cost_list = widrow_hoff_classification_ovr_ovo(X_train, y_binary, learning_rate)
        classifiers.append((w, b))
        all_epoch_lists.append(epoch_list)
        all_cost_lists.append(cost_list)

    return classifiers, all_epoch_lists, all_cost_lists


def widrow_hoff_ovr_KFold(X, y, learning_rate, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    n_classes = len(np.unique(y))
    scores = []
    all_epoch_lists = []
    all_cost_lists = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train one-vs-rest classifiers
        classifiers, epoch_lists, cost_lists = widrow_hoff_OvR(X_train, y_train, learning_rate)

        # Initialize the scores array for storing predictions for each class
        accuracy = np.zeros((X_test.shape[0], n_classes))

        # Get prediction score from each classifier
        for class_idx in range(n_classes):
            w, b = classifiers[class_idx]
            accuracy[:, class_idx] = np.dot(X_test, w) + b

            # Predict the class with the highest score for each sample
            y_predicted = np.argmax(accuracy, axis=1)

            # Calculate accuracy
            acc_per_split_for_same_seed = Measures.accuracy(y_test, y_predicted)
            scores.append(acc_per_split_for_same_seed)

            # Store epoch and cost lists for analysis
            all_epoch_lists.append(epoch_lists)
            all_cost_lists.append(cost_lists)
    acc = np.array(scores).mean()
    return acc




def widrow_hoff_OvO(X_train, y_train, learning_rate):
    classifiers = []
    all_epoch_lists = []
    all_cost_lists = []
    n_classes = len(np.unique(y_train))

    # Create all possible class pairs (one-vs-one)
    class_pairs = list(itertools.combinations(range(n_classes), 2))

    for class1, class2 in class_pairs:
        # Create binary labels for one-vs-one classification (class1 vs class2)
        y_binary = np.where(y_train == class1, 1, np.where(y_train == class2, -1, 0))
        # Remove the samples that don't belong to class1 or class2
        X_binary = X_train[y_binary != 0]
        y_binary = y_binary[y_binary != 0]

        # Train Widrow-Hoff classifier for this pair of classes
        w, b, epoch_list, cost_list = widrow_hoff_classification_ovr_ovo(X_binary, y_binary, learning_rate)
        classifiers.append((w, b))
        all_epoch_lists.append(epoch_list)
        all_cost_lists.append(cost_list)

    return classifiers, all_epoch_lists, all_cost_lists


def widrow_hoff_ovo_KFold(X, y, learning_rate, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    n_classes = len(np.unique(y))
    scores = []
    all_epoch_lists = []
    all_cost_lists = []

    # Create all possible class pairs (one-vs-one)
    class_pairs = list(itertools.combinations(range(n_classes), 2))

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train one-vs-one classifiers for each class pair
        classifiers, epoch_lists, cost_lists = widrow_hoff_OvO(X_train, y_train, learning_rate)

        # Initialize array to store predictions for each test sample
        y_predicted = np.zeros(X_test.shape[0])

        # For each test sample, get the prediction from each classifier
        for i in range(X_test.shape[0]):
            # Initialize a vote counter for each class
            votes = np.zeros(n_classes)

            # Get prediction from each classifier
            for clf_idx, clf in enumerate(classifiers):
                class_1, class_2 = class_pairs[clf_idx]  # Get the class pair for this classifier

                # Get the classifier's decision (positive or negative side of the decision boundary)
                w, b = clf  # Assuming clf is a tuple (w, b)
                decision_score = np.dot(X_test[i], w) + b

                # Predict class based on the decision boundary: class_1 if score > 0, else class_2
                predicted_class = class_1 if decision_score > 0 else class_2

                # Update the vote count for the predicted class
                votes[predicted_class] += 1

            # Assign the predicted class based on majority vote
            y_predicted[i] = np.argmax(votes)

        # Calculate accuracy
        acc_per_split_for_same_seed = Measures.accuracy(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)

        # Store epoch and cost lists for analysis
        all_epoch_lists.append(epoch_lists)
        all_cost_lists.append(cost_lists)

    acc = np.array(scores).mean()
    return acc

#############################################################################################################
def widrow_hoff_mc_ovr(X_train, y_train, learning_rate, recod_cost_at_each_epochs):
    n_classes = len(np.unique(y_train))
    n_samples, n_features = X_train.shape
    classifiers = [(np.zeros(n_features), 0) for _ in range(n_classes)]  # Initialize weights and biases
    epoch_list = []
    cost_list = []
    acc_list = []

    x_minibatch = []
    y_minibatch = []


    for i in range(n_samples):

        x_minibatch.append(X_train[i])
        y_minibatch.append(y_train[i])

        # Process each class for the current sample
        for class_idx in range(n_classes):
            w, b = classifiers[class_idx]
            # Create binary labels for one-vs-rest classification
            y_binary = 1 if y_train[i] == class_idx else -1
            x_i = X_train[i]

            # Calculate the prediction
            y_pred = np.dot(w, x_i) + b

            # Update weights and bias
            w += learning_rate * (y_binary - y_pred) * x_i
            b += learning_rate * (y_binary - y_pred)
            classifiers[class_idx] = (w, b)  # Update the classifier


        # Record cost and accuracy every `recod_cost_at_each_epochs` samples
        if i % recod_cost_at_each_epochs == 0:
            # lms_acc = Measures.widrow_hoff_acc_mc_ovr(classifiers, X_train[i], y_train[i], n_classes)
            lms_acc = Measures.widrow_hoff_acc_mc_ovr(classifiers, np.array(x_minibatch), np.array(y_minibatch), n_classes)
            acc_list.append(lms_acc)
            epoch_list.append((i+1))

            x_minibatch = []
            y_minibatch = []

    return classifiers, epoch_list, cost_list, acc_list



