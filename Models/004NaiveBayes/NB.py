import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

n_samples = 100
n_features = 2
n_classes = 2

# Create a binary classification dataset
X, y = datasets.make_classification(n_samples=n_samples,
                                    n_features=n_features,
                                    n_classes=n_classes,
                                    n_clusters_per_class=1,
                                    n_redundant=0,
                                    n_informative=2,
                                    flip_y=0.0,
                                    random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Gaussian Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = nb.predict(X_test)

# Compute the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Extract parameters of the trained Gaussian Naive Bayes model
means = nb.theta_
variances = nb.var_


# Calculate the linear decision boundary
def linear_decision_boundary(means, variances):
    # Means and variances are in the shape (n_classes, n_features)
    # Calculate the difference in means
    delta_mean = means[1] - means[0]

    # Calculate the average variance for each feature
    avg_variance = np.mean(variances, axis=0)

    # Calculate weights and intercept
    w = delta_mean / avg_variance
    b = -0.5 * np.dot(delta_mean, (means[1] + means[0]) / avg_variance)

    return w, b


# Calculate weights (w) and bias (b)
w, b = linear_decision_boundary(means, variances)

# Plot the dataset
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')

# Plot the decision boundary
x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
y_values = -(w[0] * x_values + b) / w[1]  # Compute y values for the linear boundary
plt.plot(x_values, y_values, color='green', label='Decision Boundary')

# Add labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Gaussian Naive Bayes Linear Decision Boundary')
plt.legend()
plt.show()
