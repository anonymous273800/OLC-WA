from sklearn.model_selection import KFold
from Utils import Predictions, Measures
from sklearn.naive_bayes import GaussianNB
import numpy as np

def online_naive_bayes(X, y, record_cost_on_every_epoch=50):
    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    x_minibatch = []
    y_minibatch = []

    naive_bayes_model = GaussianNB()
    # Specify the unique classes during the first call to partial_fit
    classes = np.unique(y)

    # Simulate online learning
    for i in range(1, len(y)):
        sample = X[i].reshape(1, -1)  # Reshape for sklearn
        label = y[i]  # Keep label as a scalar

        x_minibatch.append(sample)
        y_minibatch.append(label)

        # Update sklearn model
        if i == 1:  # Specify classes only on the first iteration
            naive_bayes_model.partial_fit(sample, np.array([label]), classes=classes)
        else:
            naive_bayes_model.partial_fit(sample, np.array([label]))

        # Prediction and loss calculation
        predicted_prob = naive_bayes_model.predict_proba(sample)[0]  # Get predicted probabilities
        actual_class_idx = np.where(classes == label)[0][0]  # Index of actual class in prob array
        # Negative log-likelihood as loss
        loss = -np.log(predicted_prob[actual_class_idx])  # Cross-entropy loss

        if i % record_cost_on_every_epoch == 0:  # Record the cost and epoch value every 50 iterations
            epoch_list = np.append(epoch_list, (i+1))

            # cost_list = np.append(cost_list, loss)

            # Calculate cost as the mean loss for the current minibatch
            mean_loss = np.mean([-np.log(
                naive_bayes_model.predict_proba(x.reshape(1, -1))[0][np.where(classes == y_minibatch[j])[0][0]])
                                 for j, x in enumerate(x_minibatch)]) if x_minibatch else 0

            cost_list = np.append(cost_list, mean_loss)

            x_minibatch_np = np.vstack(x_minibatch)
            y_minibatch_np = np.array(y_minibatch)
            predicted_y_test = Predictions.predict_naive_bayes1(x_minibatch_np, naive_bayes_model)
            accuracy = Measures.accuracy(y_minibatch_np, predicted_y_test)
            acc_list = np.append(acc_list, accuracy)

    return naive_bayes_model, epoch_list.astype(int), cost_list, acc_list







def online_naive_bayes_KFold(X, y, seed):
    kf = KFold(n_splits=5, random_state=seed, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        naive_bayes_model, epoch_list, cost_list, acc_list = online_naive_bayes(X_train, y_train)
        predicted_y_test = Predictions.predict_naive_bayes1(X_test, naive_bayes_model)
        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)

    acc = np.array(scores).mean()
    return acc