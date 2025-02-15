import numpy as np
from Utils import Predictions
from Utils import Measures
import itertools


def compute_acc_olr_wa_ovr(X_test, y_test, coeff_classifiers):
    predicted_y_test = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        # Get the predicted probabilities for each classifier
        probabilities = np.array([Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf) for clf in coeff_classifiers])
        predicted_y_test[i] = np.argmax(probabilities)

    acc = Measures.accuracy(y_test, predicted_y_test)

    return acc


def compute_acc_olr_wa_ovr_best_classifier_index(X_test, y_test, coeff_classifiers):
    predicted_y_test = np.zeros(X_test.shape[0])
    classifier_accuracies = []

    for clf in coeff_classifiers:
        clf_predictions = np.array(
            [Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf) for i in range(X_test.shape[0])])
        clf_predicted_y = np.argmax(clf_predictions, axis=1)
        clf_acc = Measures.accuracy(y_test, clf_predicted_y)
        classifier_accuracies.append(clf_acc)

    # Identify the best classifier based on the highest accuracy
    best_classifier_index = np.argmax(classifier_accuracies)
    best_classifier_acc = classifier_accuracies[best_classifier_index]

    # Overall accuracy for the One-vs-Rest model
    for i in range(X_test.shape[0]):
        probabilities = np.array(
            [Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf) for clf in coeff_classifiers])
        predicted_y_test[i] = np.argmax(probabilities)

    overall_acc = Measures.accuracy(y_test, predicted_y_test)

    return overall_acc, best_classifier_index, best_classifier_acc

def compute_acc_olr_wa_ovo(X_test, y_test, coeff_classifiers, n_classes):
    predicted_y_test = np.zeros(X_test.shape[0])
    # n_classes = len(np.unique(y_train))  # Determine the number of classes
    class_pairs = list(itertools.combinations(range(n_classes), 2))  # Generate class pairs

    for i in range(X_test.shape[0]):
        # Initialize a vote counter for each class
        votes = np.zeros(n_classes)

        # For each test sample, get the prediction from each classifier
        for clf_idx, clf in enumerate(coeff_classifiers):
            class_1, class_2 = class_pairs[clf_idx]  # Get the class pair for this classifier
            probability = Predictions.predict_prob_((X_test[i].reshape(1, -1)), clf)

            # For binary classification in OvO, assign the vote to the winning class
            if probability > 0.5:  # If probability > 0.5, vote for class_1
                votes[class_1] += 1
            else:  # If probability <= 0.5, vote for class_2
                votes[class_2] += 1

        # Assign the final predicted class based on the highest vote
        predicted_y_test[i] = np.argmax(votes)

    acc = Measures.accuracy(y_test, predicted_y_test)
    return acc