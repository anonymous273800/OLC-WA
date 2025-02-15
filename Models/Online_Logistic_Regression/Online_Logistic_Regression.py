from sklearn.model_selection import KFold
from Utils import Predictions, Measures
import numpy as np
from sklearn.linear_model import SGDClassifier
from itertools import combinations




def sigmoid(z):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-z))

# ignored since the performance was not good as the custom one.
def online_logistic_regression_sklearn_version(X_train, y_train, seed):
    n_samples, n_features = X_train.shape
    # online_logistic_regression_model = SGDClassifier(loss='log_loss', learning_rate='optimal')
    online_logistic_regression_model = SGDClassifier(max_iter=1, loss='log_loss', learning_rate='optimal',
                                                     penalty=None, tol=None, warm_start=True, random_state=seed)

    # Classes should be specified for partial_fit
    classes = np.unique(y_train)

    # Online learning: Fit one sample at a time using partial_fit
    for i in range(n_samples):
        xi = X_train[i].reshape(1, -1)  # Reshape to (1, n_features)
        yi = np.array(y_train[i]).reshape(1, )  # Ensure yi is a 1D array with shape (1,)
        # online_logistic_regression_model.partial_fit(X_train[i:i+1], y_train[i:i+1], classes=classes)
        online_logistic_regression_model.partial_fit(xi, yi, classes=classes)

    # Retrieve the model's weights and bias
    w = online_logistic_regression_model.coef_.flatten()
    b = online_logistic_regression_model.intercept_[0]

    # No need for epoch list in online setting
    cost_list = []  # You can implement cost calculations if needed, but SGDClassifier doesn't provide it directly.
    epoch_list = []
    return w, b, epoch_list, cost_list

def online_logistic_regression(X_train, y_train, learning_rate, record_cost_on_every_epoch=50):

    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    cost_list = []
    epoch_list = []
    acc_list = []

    x_minibatch = []
    y_minibatch = []

    for i in range(1, n_samples):
        xi = X_train[i]
        yi = y_train[i]

        x_minibatch.append(xi)
        y_minibatch.append(yi)

        # Prediction using the logistic function
        linear_output = np.dot(xi, w) + b
        y_pred = sigmoid(linear_output)

        # Gradient descent update
        error = y_pred - yi
        dw = error * xi
        db = error

        w -= learning_rate * dw
        b -= learning_rate * db

        # Record cumulative cost and epoch values at intervals (every 50 iterations)
        if i % record_cost_on_every_epoch == 0:
            epoch_list = np.append(epoch_list, i)

            # Convert minibatch to numpy arrays for easier computation
            x_minibatch_np = np.array(x_minibatch)
            y_minibatch_np = np.array(y_minibatch)

            linear_output_minibatch = np.dot(x_minibatch_np, w) + b
            y_pred_minibatch = sigmoid(linear_output_minibatch)

            # Calculate cost for the minibatch (mean cost)
            cost = - np.mean(y_minibatch_np * np.log(y_pred_minibatch + 1e-15) +
                             (1 - y_minibatch_np) * np.log(1 - y_pred_minibatch + 1e-15))
            cost_list.append(cost)  # Append mean cost to cost_list

            # cost = - (yi * np.log(y_pred + 1e-15) + (1 - yi) * np.log(1 - y_pred + 1e-15)) # this for a single point
            # cost_list = np.append(cost_list, cost)

            # Calculate accuracy for the minibatch
            y_pred_labels = (y_pred_minibatch >= 0.5).astype(int)  # Convert probabilities to class labels
            acc = Measures.accuracy(y_minibatch_np, y_pred_labels)
            acc_list.append(acc)  # Append accuracy to acc_list


            # Reset minibatch for the next interval
            x_minibatch = []
            y_minibatch = []


    return w, b, epoch_list, cost_list, acc_list


def online_logistic_regression_KFold(X, y, learning_rate,  seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b, epoch_list, cost_list, acc_list = online_logistic_regression(X_train, y_train, learning_rate)
        # w, b, epoch_list, cost_list = online_logistic_regression_sklearn_version(X_train, y_train, seed)
        y_predicted = Predictions.predict(X_test, w, b)
        acc_per_split_for_same_seed = Measures.accuracy(y_test, y_predicted)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()
    return acc
#################################################OVR###############################################################


def online_logistic_regression_ovr_ovo(X_train, y_train, learning_rate, record_cost_on_every_epoch):
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0
    cost_list = []
    epoch_list = []
    acc_list = []

    x_minibatch = []
    y_minibatch = []

    for i in range(n_samples):
        xi = X_train[i]
        yi = y_train[i]

        x_minibatch.append(xi)
        y_minibatch.append(yi)

        # Prediction using the logistic function
        linear_output = np.dot(xi, w) + b
        y_pred = sigmoid(linear_output)

        # Gradient descent update
        error = y_pred - yi
        dw = error * xi
        db = error

        w -= learning_rate * dw
        b -= learning_rate * db

        # Record cumulative cost and epoch values at intervals (every 50 iterations)
        if i % record_cost_on_every_epoch == 0:
            epoch_list = np.append(epoch_list, i)

            # Convert minibatch to numpy arrays for easier computation
            x_minibatch_np = np.array(x_minibatch)
            y_minibatch_np = np.array(y_minibatch)

            linear_output_minibatch = np.dot(x_minibatch_np, w) + b
            y_pred_minibatch = sigmoid(linear_output_minibatch)

            # Calculate cost for the minibatch (mean cost)
            cost = - np.mean(y_minibatch_np * np.log(y_pred_minibatch + 1e-15) +
                             (1 - y_minibatch_np) * np.log(1 - y_pred_minibatch + 1e-15))
            cost_list.append(cost)  # Append mean cost to cost_list

            # cost = - (yi * np.log(y_pred + 1e-15) + (1 - yi) * np.log(1 - y_pred + 1e-15)) # this for a single point
            # cost_list = np.append(cost_list, cost)

            # Calculate accuracy for the minibatch
            y_pred_labels = (y_pred_minibatch >= 0.5).astype(int)  # Convert probabilities to class labels
            acc = Measures.accuracy(y_minibatch_np, y_pred_labels)
            acc_list.append(acc)  # Append accuracy to acc_list

            # Reset minibatch for the next interval
            x_minibatch = []
            y_minibatch = []

    return w, b, epoch_list, cost_list, acc_list

def online_logistic_regression_ovr(X_train, y_train, learning_rate, record_cost_on_every_epoch):
    n_classes = len(np.unique(y_train))
    classifiers = []
    all_epoch_lists = []
    all_cost_lists = []
    all_acc_lists = []
    for class_idx in range(n_classes):
        # Create binary labels for one-vs-rest classification
        y_binary = np.where(y_train == class_idx, 1, 0)

        w, b, epoch_list, cost_list, acc_list = online_logistic_regression_ovr_ovo(X_train, y_binary, learning_rate, record_cost_on_every_epoch)
        classifiers.append((w, b))
        all_epoch_lists.append(epoch_list)
        all_cost_lists.append(cost_list)
        all_acc_lists.append(acc_list)


    return classifiers, all_epoch_lists, all_cost_lists, all_acc_lists

def online_logistic_regression_ovr_KFold(X, y, learning_rate, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    # n_classes = len(np.unique(y))
    scores = []
    all_epoch_lists = []
    all_cost_lists = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train one-vs-rest classifiers
        classifiers, epoch_lists, cost_lists = online_logistic_regression_ovr(X_train, y_train, learning_rate)

        predicted_y_test = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            # Get the predicted probabilities for each classifier
            probabilities = np.array(
                [Predictions.predict_prob_2((X_test[i].reshape(1, -1)), clf[0], clf[1]) for clf in classifiers])
            predicted_y_test[i] = np.argmax(probabilities)

        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

        # Store epoch and cost lists for analysis
        all_epoch_lists.append(epoch_lists)
        all_cost_lists.append(cost_lists)

    return np.array(scores).mean()
########################################OVO#######################################################################


def online_logistic_regression_ovo(X_train, y_train, learning_rate):
    n_classes = len(np.unique(y_train))
    classifiers = []
    all_epoch_lists = []
    all_cost_lists = []
    unique_classes = np.unique(y_train)
    class_pairs = list(combinations(unique_classes, 2))  # Create all possible pairs of classes

    # Generate all combinations of classes
    for class_pair in class_pairs:
        class_1, class_2 = class_pair
        # Prepare binary labels for class 1 vs class 2
        mask = np.isin(y_train, [class_1, class_2])  # Filter out samples belonging to class_1 or class_2
        binary_X_train = X_train[mask]  # Select only relevant samples
        binary_y_train = np.where(y_train[mask] == class_1, 1, 0)  # Assign 1 for class_1, 0 for class_2

        # Train the binary classifier using online logistic regression
        w, b, epoch_list, cost_list = online_logistic_regression_ovr_ovo(binary_X_train, binary_y_train, learning_rate)

        # Append the classifier for the current pair
        classifiers.append((w, b))
        all_epoch_lists.append(epoch_list)
        all_cost_lists.append(cost_list)

    return classifiers, all_epoch_lists, all_cost_lists

def online_logistic_regression_ovo_KFold(X, y, learning_rate, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    all_epoch_lists = []
    all_cost_lists = []
    classes = len(np.unique(y))
    unique_classes = np.unique(y)
    class_pairs = list(combinations(unique_classes, 2))  # Create all possible pairs of classes
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Train one-vs-rest classifiers
        classifiers, epoch_lists, cost_lists = online_logistic_regression_ovo(X_train, y_train, learning_rate)

        # Initialize an array to hold votes for each class
        votes = np.zeros((X_test.shape[0], classes))


        # For each test sample, gather votes from all classifiers
        for i in range(X_test.shape[0]):
            for idx, (w, b) in enumerate(classifiers):
                # Predict the output using logistic regression (sigmoid function)
                prob = Predictions.predict_prob_2(X_test[i], w, b)
                predicted_class = 1 if prob > 0.5 else -1  # Binary classification result

                # Get the class pair that corresponds to the current classifier
                class_1, class_2 = class_pairs[idx]

                # Update the votes: If predicted_class == 1, vote for class_1, else vote for class_2
                if predicted_class == 1:
                    votes[i, class_1] += 1
                else:
                    votes[i, class_2] += 1

        # Predict the class with the most votes for each test sample
        predicted_y_test = np.argmax(votes, axis=1)

        acc = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc)

        # Store epoch and cost lists for analysis
        all_epoch_lists.append(epoch_lists)
        all_cost_lists.append(cost_lists)

    return np.array(scores).mean()

#########################################################################################

# Function for Online Logistic Regression (One-vs-Rest)
def online_logistic_regression_ovr_final(X_train, y_train, learning_rate, record_cost_on_every_epoch):
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    classifiers = [(np.zeros(n_features), 0) for _ in range(n_classes)]  # Initialize weights and biases for all classifiers
    epoch_list = []
    acc_list = []

    # Initialize minibatches for recording
    minibatch_x = []
    minibatch_y = []

    # Loop over samples
    for i in range(n_samples):
        xi = X_train[i]
        yi = y_train[i]

        # Train each classifier (One-vs-Rest)
        for class_idx in range(n_classes):
            # Binary labels for this classifier
            y_binary = 1 if yi == class_idx else 0

            # Unpack weights and bias
            w, b = classifiers[class_idx]

            # Prediction using logistic function
            linear_output = np.dot(xi, w) + b
            y_pred = sigmoid(linear_output)

            # Gradient descent update
            error = y_pred - y_binary
            dw = error * xi
            db = error

            # Update weights and bias
            w -= learning_rate * dw
            b -= learning_rate * db

            # Store updated weights and bias
            classifiers[class_idx] = (w, b)

        # Append to minibatch
        minibatch_x.append(xi)
        minibatch_y.append(yi)

        # Record accuracy at intervals
        if i % record_cost_on_every_epoch == 0:
            acc = Measures.olr_acc_mc_ovr(classifiers, np.array(minibatch_x), np.array(minibatch_y))
            acc_list.append(acc)
            epoch_list.append(i + 1)

            # Reset minibatches
            minibatch_x = []
            minibatch_y = []

    return classifiers, epoch_list, None, acc_list