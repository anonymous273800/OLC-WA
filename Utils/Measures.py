import numpy as np


def accuracy(y_test, y_pred):
    """Compute accuracy as the percentage of correct predictions."""
    # print('y_test', y_test)
    # print('y_pred',y_pred)
    # print(y_test.shape, y_pred.shape)
    # print("types: ",y_test.dtype, y_pred.dtype)
    # print("isNan:", np.isnan(y_test).sum(), np.isnan(y_pred).sum())
    # print("acc: ", np.mean(y_test == y_pred))
    # print('------------------------------------------')
    y_test = np.array(y_test, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    return np.mean(y_test == y_pred)


def cross_entopy_cost(y, y_pred):
    # Compute binary cross-entropy loss
    n = y.shape[0]  # number of samples
    cost = -1 / n * np.sum(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    return cost


# def widrow_hoff_acc_mc_ovr(lms_classifiers, X_test, y_test, n_classes):
#     lms_scores = []
#     accuracy_ = np.zeros((X_test.shape[0], n_classes))
#     # Get prediction score from each classifier
#     for class_idx in range(n_classes):
#         w, b = lms_classifiers[class_idx]
#         accuracy_[:, class_idx] = np.dot(X_test, w) + b
#         # Predict the class with the highest score for each sample
#         y_predicted = np.argmax(accuracy_, axis=1)
#         lms_ac = accuracy(y_test, y_predicted)
#         lms_scores.append(lms_ac)
#     lms_acc = np.array(lms_scores).mean()
#     return lms_acc

def widrow_hoff_acc_mc_ovr(lms_classifiers, X_test, y_test, n_classes):
    # Initialize a matrix to store the prediction scores for each sample and each class
    scores = np.zeros((X_test.shape[0], n_classes))

    # Compute prediction scores from each classifier in a one-vs-rest setup
    for class_idx in range(n_classes):
        w, b = lms_classifiers[class_idx]
        # For each class, calculate the score as a linear combination of weights and inputs
        scores[:, class_idx] = np.dot(X_test, w) + b

        # Determine the predicted class for each sample by selecting the highest score across classes
    y_predicted = np.argmax(scores, axis=1)

    # Calculate the accuracy based on overall predictions across all samples
    lms_acc = accuracy(y_test, y_predicted)
    return lms_acc


def perceptron_acc_mc_ovr(classifiers, X_test, y_test):
    # Predict using the one-vs-rest classifiers
    y_pred = []
    for i in range(len(X_test)):
        scores = np.array([np.dot(X_test[i], w) + b for w, b in classifiers])
        predicted_class = np.argmax(scores)  # Choose the class with the highest score
        y_pred.append(predicted_class)
        # Calculate accuracy
    acc = accuracy(y_test, y_pred)
    return acc


from Utils import Predictions


def olr_acc_mc_ovr(classifiers, X_test, y_test):
    predicted_y_test = np.zeros(X_test.shape[0], dtype=int)  # Ensure integer type for predicted labels
    for i in range(X_test.shape[0]):
        # Get the predicted probabilities for each classifier
        probabilities = np.array(
            [Predictions.predict_prob_2((X_test[i].reshape(1, -1)), clf[0], clf[1]) for clf in classifiers]
        )
        predicted_y_test[i] = np.argmax(probabilities)

    # Assuming accuracy is defined in Measures
    online_logistic_regression_acc = accuracy(y_test, predicted_y_test)

    return online_logistic_regression_acc

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
#     acc = accuracy(np.array(y_minibatch), np.array(y_pred_batch))
#     return acc