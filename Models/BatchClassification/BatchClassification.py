from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from Utils import Predictions, Measures

def logistic_regression(xs, ys):
    # model = LogisticRegression(solver='liblinear')
    model = LogisticRegression()
    # model = LogisticRegression(solver='saga')
    model.fit(xs, ys)
    w = model.coef_[0]
    b = model.intercept_[0]
    return w, b


def logistic_regression_OLC_WA(xs, ys):
    model = LogisticRegression(solver='liblinear')
    # model = LogisticRegression(solver='saga')
    model.fit(xs, ys)

    w = model.coef_[0]
    b = model.intercept_[0]
    return w, b

from sklearn.metrics import accuracy_score
def logistic_regression_OLC_WA2(xs, ys):
    model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state=42)
    model.fit(xs, ys)
    # Predict on the test set
    y_pred = model.predict(xs)

    # Evaluate the model
    accuracy = accuracy_score(ys, y_pred)
    print(f"Accuracy on the XXXX set: {accuracy:.2f}")

    w = model.coef_
    b = model.intercept_
    return w, b

# from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
# def PA_OLC_WA(xs, ys):
#     w, b, cost_list, epoch_list = OnlinePassiveAggressive.online_passive_aggressive_sklearn(xs,ys,C=.1,seed=42)
#     return w, b

def plot_decison_boundary(w, b, X, y):
    # Generate x1 values (e.g., between -1 and 1)
    x1_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    # Calculate corresponding x2 values
    x2_vals = decision_boundary_x2(w, b, x1_vals)

    # Plot the decision boundary
    plt.plot(x1_vals, x2_vals, label='Decision Boundary', color='black')

    # # Optionally, scatter some data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', label='Data')

    plt.xlabel('Feature 1 (x1)')
    plt.ylabel('Feature 2 (x2)')
    plt.legend()
    plt.title('Decision Boundary Plot')
    plt.show()


def predict(x_test, w, b):
    """Predict class labels based on the learned weights and intercept."""
    # Compute the logit (linear combination of features and weights)
    z = np.dot(x_test, w) + b

    # Apply the sigmoid function to get probabilities
    probs = sigmoid(z)

    # Convert probabilities to class labels (threshold at 0.5)
    y_pred = (probs >= 0.5).astype(int)

    return y_pred


def decision_boundary_x2(w,b, x1):
    return -(b + w[0] * x1) / w[1]

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_accuracy(y_pred, y_test):
    """Compute accuracy as the percentage of correct predictions."""
    return np.mean(y_test == y_pred)


def batch_logistic_regression_KFold(X, y, seed, shuffle):
    kf = KFold(n_splits=5, random_state=seed, shuffle=shuffle)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        w, b = logistic_regression(X_train, y_train)
        predicted_y_test = Predictions.predict(X_test, w, b)
        acc_per_split_for_same_seed = Measures.accuracy(y_test, predicted_y_test)
        scores.append(acc_per_split_for_same_seed)
    acc = np.array(scores).mean()

    return acc





# if __name__ == "__main__":
#     random_state = 42
#     X, y = DS01.get_DS01()
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_state)
#     w, b = logistic_regression(X_train, y_train)
#     y_pred = predict(X_test, w,b)
#     accuracy = compute_accuracy(y_pred, y_test)
#     print('accuracy', accuracy)
#     plot_decison_boundary(w,b, X_test, y_test)



