import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def widrow_hoff(X_train, y_train):
    learning_rate = 0.01
    n_samples, n_features = X_train.shape
    w = np.zeros(n_features)
    b = 0

    # Convert labels to {-1, 1} for binary classification
    y_train = np.where(y_train == 0, -1, 1)

    for i in range(n_samples):
        x_i = X_train[i]
        y_i = y_train[i]

        y_pred = np.dot(w, x_i) + b
        print('y_pred: ', y_pred)

        error = y_i - y_pred
        print('error: ',error)

        w = w + learning_rate * error * x_i
        b = b + learning_rate * error
    return w,b







def predict(X, w, b):
    # Predict using the weight vector and bias
    linear_output = np.dot(X, w) + b
    return np.where(linear_output >= 0, 1, 0)  # return 1 if output >= 0 else 0


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

if __name__ == "__main__":
    n_samples = 10000
    n_features = 2
    n_classes = 2
    n_clusters_per_class = 1
    n_redundant = 0
    n_informative = 2
    flip_y = 0.0
    class_sep = 1.0
    shuffle = True
    random_state = 42
    X, y = datasets.make_classification(n_samples=n_samples,
                                        n_features=n_features,
                                        n_classes=n_classes,
                                        n_clusters_per_class=n_clusters_per_class,
                                        n_redundant=n_redundant,
                                        n_informative=n_informative,
                                        flip_y=flip_y,
                                        class_sep=class_sep,
                                        shuffle=shuffle,
                                        random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
    w,b = widrow_hoff(X_train, y_train)

    # linear_output = np.dot(X_test, w) + b  # []
    # y_predict = np.where(linear_output >= 0, 1, 0)
    y_predict = predict(X_test, w, b)
    accuracy = compute_accuracy(y_test, y_predict)
    print(f"Accuracy: {accuracy * 100:.2f}%")




