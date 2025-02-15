import numpy as np


def predict(x_test, w, b):
    """Predict class labels based on the learned weights and intercept."""
    # Compute the logit (linear combination of features and weights)
    # z = np.dot(x_test, w) + b

    # # Compute the logit (linear combination of features and weights)
    # if isinstance(x_test, np.ndarray):  # If dense, use np.dot
    #     z = np.dot(x_test, w) + b
    # else:  # If sparse, use sparse matrix dot

    z = x_test.dot(w) + b

    z = np.asarray(z)  # Ensure z is a NumPy array
    # print('z', z)

    # Apply the sigmoid function to get probabilities
    probs = sigmoid(z)

    # Convert probabilities to class labels (threshold at 0.5)
    y_pred = (probs >= 0.5).astype(int)

    return y_pred


# def predict(x_test, w, b):
#     """Predict class labels based on the learned weights and intercept."""
#     # Compute the logit (linear combination of features and weights)
#     z = np.dot(x_test, w) + b
#     print("z after dot product:", z)  # Debugging statement
#     print("Type of z:", type(z))  # Debugging statement
#     print("Shape of z:", z.shape)  # Debugging statement
#
#     z = np.asarray(z)  # Ensure z is a NumPy array
#     print("Calling sigmoid with z:", z)  # Debugging statement
#
#     # Apply the sigmoid function to get probabilities
#     probs = sigmoid(z)
#
#     # Convert probabilities to class labels (threshold at 0.5)
#     y_pred = (probs >= 0.5).astype(int)
#
#     return y_pred


def predict_(x_test, base_coeff):
    """
    Predict class labels based on the learned weights and intercept,
    extracting parameters from base_coeff.

    Args:
        x_test (numpy array): The feature matrix for testing.
        base_coeff (numpy array): The coefficients array containing weights,
                                  the bias term, and a scaling factor.

    Returns:
        numpy array: Predicted class labels (0 or 1).
    """
    # Extract weights, bias, and scaling factor from base_coeff
    d_b = base_coeff[-1]  # Bias term (last element)
    c_b = base_coeff[-2]  # Scaling factor (second to last element)
    w = base_coeff[:-2]  # Weights (all except the last two elements)

    # Compute the logit (linear combination of features and weights)
    # z = np.dot(x_test, w) + d_b
    if isinstance(x_test, np.ndarray):  # If dense, use np.dot
        z = np.dot(x_test, w) + d_b
    else:  # If sparse, use sparse matrix dot
        z = x_test.dot(w) + d_b

    # Apply the sigmoid function to get probabilities
    probs = sigmoid(z)  # / c_b  # Scale the probabilities

    # Convert probabilities to class labels (threshold at 0.5)
    y_pred = (probs >= 0.5).astype(int)

    return y_pred




def predict_XXX(x_test, base_coeff):
    """
    Predict class labels based on the learned weights and intercept,
    extracting parameters from base_coeff.

    Args:
        x_test (numpy array): The feature matrix for testing.
        base_coeff (numpy array): The coefficients array containing weights,
                                  the bias term, and a scaling factor.

    Returns:
        numpy array: Predicted class labels (0 or 1).
    """
    # Extract weights, bias, and scaling factor from base_coeff
    d_b = base_coeff[-1]  # Bias term (last element)
    c_b = base_coeff[-2]  # Scaling factor (second to last element)
    w = base_coeff[:-2]  # Weights (all except the last two elements)

    # Compute the logit (linear combination of features and weights)
    z = np.dot(x_test, w) + d_b

    # Apply the sigmoid function to get probabilities
    probs = sigmoid(z)  # / c_b  # Scale the probabilities

    # Convert probabilities to class labels (threshold at 0.5)
    y_pred = (probs >= 0.5).astype(int)

    return y_pred


# def _compute_predictions__(xs, w):
#
#     d_b = w[-1]
#     c_b = w[-2]
#     w = w[0:-2]
#     z = np.array([(-1 * (np.dot(w, x) + d_b) ) for x in xs]).flatten()
#     # Apply the sigmoid function to get probabilities
#     probs = sigmoid(z)  # / c_b  # Scale the probabilities
#     # Convert probabilities to class labels (threshold at 0.5)
#     y_pred = (probs >= 0.5).astype(int)
#     return y_pred

# the w order is where b is the first element and the rest is w
def predict_2(x_test, base_coeff):
    # Extract weights, bias, and scaling factor from base_coeff
    d_b = base_coeff[0]  # Bias term (last element)
    w = base_coeff[1:]  # Weights (all except the last two elements)

    # Compute the logit (linear combination of features and weights)
    z = np.dot(x_test, w) + d_b

    # Apply the sigmoid function to get probabilities
    probs = sigmoid(z)  # / c_b  # Scale the probabilities

    # Convert probabilities to class labels (threshold at 0.5)
    y_pred = (probs >= 0.5).astype(int)

    return y_pred


# def predict_3(X_test, w):
#     # Add a bias term (column of ones) to X_test
#     x0 = np.ones(len(X_test))
#     X_test = np.concatenate((np.matrix(x0).T, X_test), axis=1)
#
#     # Compute the raw predictions (dot product with the weight vector)
#     raw_predictions = np.dot(X_test, w)
#
#     # Apply threshold to get class labels (1 or -1)
#     y_predicted = np.where(raw_predictions >= 0, 1, -1)
#
#     return y_predicted


def predict_prob_(x_test, base_coeff):
    """
    Predict class labels based on the learned weights and intercept,
    extracting parameters from base_coeff.

    Args:
        x_test (numpy array): The feature matrix for testing.
        base_coeff (numpy array): The coefficients array containing weights,
                                  the bias term, and a scaling factor.

    Returns:
        numpy array: Predicted class labels (0 or 1).
    """
    # Extract weights, bias, and scaling factor from base_coeff
    d_b = base_coeff[-1]  # Bias term (last element)
    c_b = base_coeff[-2]  # Scaling factor (second to last element)
    w = base_coeff[:-2]  # Weights (all except the last two elements)

    # Compute the logit (linear combination of features and weights)
    z = np.dot(x_test, w) + d_b

    # Apply the sigmoid function to get probabilities
    probs = sigmoid(z)  # / c_b  # Scale the probabilities

    return probs


def sigmoid(z):
    z = np.array(z, dtype=float)  # Ensure z is a NumPy array
    return 1 / (1 + np.exp(-z))


def predict_multiclass(X, w, b):
    scores = np.dot(X, w.T) + b  # w.T to handle (n_classes, n_features) * (n_features, n_samples)
    probabilities = softmax(scores, axis=1)
    predicted_classes = np.argmax(probabilities, axis=1)
    return predicted_classes


def softmax(z, axis=1):
    exp_z = np.exp(z - np.max(z, axis=axis, keepdims=True))
    return exp_z / np.sum(exp_z, axis=axis, keepdims=True)


def predict_multiclass2(X_test, w, b):
    # Compute the decision function for all classes (dot product of X and w, plus the bias)
    z = np.dot(X_test, w.T) + b  # Transpose w to align shapes (200, 2) dot (3, 2).T = (200, 3)

    # Predict the class with the highest score for each sample
    predicted_y = np.argmax(z, axis=1)  # Returns the index of the maximum score along each row

    return predicted_y


def predict_widrow_hoff(x_test, w, b):
    # Predict using the weight vector and bias
    linear_output = np.dot(x_test, w) + b
    return np.where(linear_output >= 0, 1, 0)  # return 1 if output >= 0 else 0


def predict_widrow_hoff2(x_test, w, b):
    # Predict using the weight vector and bias
    linear_output = np.dot(x_test, w) + b
    return np.where(linear_output >= 0, 1, 0)  # return 1 if output >= 0 else 0


def predict_perceptron(x_test, w, b):
    linear_model = np.dot(x_test, w) + b
    y_predicted = np.sign(linear_model)
    return y_predicted


def predict_naive_bayes1(x_test, naive_base_model):
    predicted_y_test = naive_base_model.predict(x_test)
    return predicted_y_test

def predict_hoefding_tree(x_test, model):
    predicted_y_test = []
    for i in range(x_test.shape[0]):
        x_test_instance = {f'feature_{j}': x_test[i, j] for j in range(x_test.shape[1])}
        predicted_y = model.predict_one(x_test_instance)
        predicted_y_test.append(predicted_y)
    return np.array(predicted_y_test)


def predict_prob_2(X, w, b):
    linear_output = np.dot(X, w) + b
    probabilities = sigmoid(linear_output)
    return probabilities



def predict_hoeffding_tree(X_test, model):
    n_samples, n_features = X_test.shape
    y_predicted = []
    for i in range(len(X_test)):
        x_test_instance = {f'feature_{j}': X_test[i, j] for j in range(n_features)}
        y_test_pred = model.predict_one(x_test_instance)
        y_predicted.append(y_test_pred)

    return y_predicted