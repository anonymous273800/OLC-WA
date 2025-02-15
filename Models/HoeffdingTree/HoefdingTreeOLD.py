import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from river import tree
from river import metrics

# Create a synthetic dataset
n_samples = 10000
n_features = 20
n_classes = 2  # For binary classification

X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_classes=n_classes, n_informative=10,
                           n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Hoeffding Tree classifier
model = tree.HoeffdingTreeClassifier()

# # Create a metric to evaluate performance
# train_metric = metrics.Accuracy()

# Train the model using the training data
for i in range(len(X_train)):
    # Create a dictionary for the current instance
    x_instance = {f'feature_{j}': X_train[i, j] for j in range(n_features)}

    # Update the model with the current instance
    model.learn_one(x_instance, y_train[i])

    # # Make a prediction
    # y_pred = model.predict_one(x_instance)

    # Update the accuracy metric
    # train_metric = train_metric.update(y_train[i], y_pred)  # Reassign the updated metric

# # Output the final training accuracy
# print(f'Training Accuracy: {train_metric:.4f}')

y_predicted = []

for i in range(len(X_test)):
    x_test_instance = {f'feature_{j}': X_test[i, j] for j in range(n_features)}
    y_test_pred = model.predict_one(x_test_instance)
    y_predicted.append(y_test_pred)

from Utils import Measures
acc = Measures.accuracy(y_test, y_predicted)
print(acc)

print('###############################################')

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class HoeffdingTreeNode:
    def __init__(self):
        self.is_leaf = True
        self.class_counts = None  # Class counts for the leaf node
        self.split_feature = None  # Feature index to split on
        self.split_value = None  # Value to split on
        self.children = []  # Children nodes
        self.feature_values = []  # To store unique feature values observed


class HoeffdingTree:
    def __init__(self, delta=1e-5, min_samples_split=100):
        self.root = HoeffdingTreeNode()
        self.delta = delta  # Hoeffding bound delta
        self.min_samples_split = min_samples_split
        self.sample_count = 0  # Total number of samples seen

    def learn_one(self, x, y):
        # Increment the total sample count
        self.sample_count += 1
        self._update_node(self.root, x, y)

    def _update_node(self, node, x, y):
        if node.is_leaf:
            # If this is the first sample, initialize class counts
            if node.class_counts is None:
                node.class_counts = np.zeros(2)  # Assuming binary classification
            node.class_counts[y] += 1

            # Store the observed feature values
            for feature in range(len(x)):
                if feature >= len(node.feature_values):
                    node.feature_values.append(set())  # Initialize set for new feature
                node.feature_values[feature].add(x[feature])

            # If we have enough samples, try to split
            if self.sample_count >= self.min_samples_split:
                self._try_split(node)

        else:
            # Follow the tree down to the correct child
            if x[node.split_feature] <= node.split_value:
                self._update_node(node.children[0], x, y)
            else:
                self._update_node(node.children[1], x, y)

    def _try_split(self, node):
        # Example split criteria based on Gini impurity or information gain
        best_feature = None
        best_value = None
        best_gain = -1

        for feature, values in enumerate(node.feature_values):  # Loop over features
            for value in values:  # Unique values observed for this feature
                gain = self._calculate_gain(node, feature, value)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_value = value

        # If a good split is found, create new child nodes
        if best_gain > 0:  # Arbitrary threshold for split gain
            node.is_leaf = False
            node.split_feature = best_feature
            node.split_value = best_value
            node.children = [HoeffdingTreeNode(), HoeffdingTreeNode()]

    def _calculate_gain(self, node, feature, value):
        # This function would calculate the gain from splitting
        # For now, we'll return a placeholder value
        return np.random.rand()  # Replace with actual gain calculation

    def predict_one(self, x):
        return self._predict_node(self.root, x)

    def _predict_node(self, node, x):
        if node.is_leaf:
            return np.argmax(node.class_counts)
        else:
            if x[node.split_feature] <= node.split_value:
                return self._predict_node(node.children[0], x)
            else:
                return self._predict_node(node.children[1], x)


# Create a synthetic dataset
n_samples = 10000
n_features = 20
n_classes = 2  # For binary classification

X, y = make_classification(n_samples=n_samples, n_features=n_features,
                           n_classes=n_classes, n_informative=10,
                           n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Hoeffding Tree classifier
model = HoeffdingTree()

# Train the model using the training data
for i in range(len(X_train)):
    model.learn_one(X_train[i], y_train[i])

# Prepare predictions for the test set
y_predicted = []
for i in range(len(X_test)):
    y_test_pred = model.predict_one(X_test[i])
    y_predicted.append(y_test_pred)

# Calculate accuracy
accuracy = np.mean(np.array(y_test) == np.array(y_predicted))
print(f'Testing Accuracy: {accuracy:.4f}')

