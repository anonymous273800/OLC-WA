from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from Datasets import DS01

def LogisticRegression_liblinear_solver_fn():
    random_state = 42

    X, y = DS01.get_DS01()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=random_state)
    model = LogisticRegression(solver='sag')
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Plot the dataset
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

    # Plot decision boundary
    coef = model.coef_[0]
    intercept = model.intercept_
    xx = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yy = -(coef[0] * xx + intercept) / coef[1]
    plt.plot(xx, yy, 'k--', label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Classification Dataset with Decision Boundary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    LogisticRegression_liblinear_solver_fn()