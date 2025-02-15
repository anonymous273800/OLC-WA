import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

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

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the ANN model (linear model)
model = Sequential()

# Input layer and output layer with 1 neuron (logistic regression)
model.add(Dense(1, input_dim=n_features, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Accuracy on test set: {accuracy:.2f}')

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Calculate accuracy
print(f'Accuracy Score: {accuracy_score(y_test, y_pred):.2f}')

# Plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Use the model to predict across the grid of values
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = (Z > 0.5).astype(int)
Z = Z.reshape(xx.shape)

# Plot the contour and training points
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', label="Training Data")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='x', label="Test Data")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.legend()
plt.title('Linear Decision Boundary and Data Points')
plt.show()
