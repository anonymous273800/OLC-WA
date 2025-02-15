import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from Utils import Util, DriftDatasetsCharacteristics

'''
DS17
Drift Type: Abrupt - multiclass
Total n_samples: 1500 (3 classes each has 250 + drifted 3 classes each 250)
Dimensions: 2
Drift Location: Point 750
'''

def get_DS17():
    np.random.seed(42)
    n_samples_per_class = 250
    drift_location = 750
    drift_type = 'Abrupt'

    #DSA
    mean_class_0 = np.array([0.0 , 0.0]) # Mean of class 0
    mean_class_1 = np.array([3.0, 3.0]) # Mean of class 1
    mean_class_2 = np.array([3.0, -2.0])  # Mean of class 2
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])  # Same covariance for both classes

    # Generate original dataset for both classes
    X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
    X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)
    X_class_2 = np.random.multivariate_normal(mean_class_2, covariance, n_samples_per_class)

    # Labels for each class
    y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
    y_class_1 = np.ones(n_samples_per_class)  # Class 1 label
    y_class_2 = np.full(n_samples_per_class, 2)    # Class 2 label

    # Combine class 0 and class 1 datasets
    X_original = np.vstack((X_class_0, X_class_1, X_class_2))
    y_original = np.hstack((y_class_0, y_class_1, y_class_2))

    # shuffle original dataset
    X_original, y_original = Util.shuffle_dataset(X_original, y_original)

    # Parameters for the drifted dataset (shift the mean of one or both classes)
    mean_class_0_drifted = np.array([10.0, 10.0])  # Abrupt new mean for class 0 after drift
    mean_class_1_drifted = np.array([15.0, 15.0])  # Abrupt new mean for class 1 after drift
    mean_class_2_drifted = np.array([15.0, 10.0])  # Abrupt new mean for class 1 after drift

    # Generate drifted dataset
    X_class_0_drifted = np.random.multivariate_normal(mean_class_0_drifted, covariance, n_samples_per_class)
    X_class_1_drifted = np.random.multivariate_normal(mean_class_1_drifted, covariance, n_samples_per_class)
    X_class_2_drifted = np.random.multivariate_normal(mean_class_2_drifted, covariance, n_samples_per_class)

    # Combine drifted class 0 and class 1 datasets
    X_drifted = np.vstack((X_class_0_drifted, X_class_1_drifted, X_class_2_drifted))
    y_drifted = np.hstack((y_class_0, y_class_1, y_class_2))  # Labels remain the same

    #Shuffle Drifted Dataset
    X_drifted, y_drifted = Util.shuffle_dataset(X_drifted, y_drifted)


    # Combine the original and drifted datasets
    X_combined = np.vstack((X_original, X_drifted))
    y_combined = np.hstack((y_original, y_drifted))

    # Plot original and drifted datasets
    plt.scatter(X_class_0[:, 0], X_class_0[:, 1], color='blue', label='Class 0 (Original)', alpha=0.5)
    plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='green', label='Class 1 (Original)', alpha=0.5)
    plt.scatter(X_class_2[:, 0], X_class_2[:, 1], color='purple', label='Class 2 (Original)', alpha=0.5)
    plt.scatter(X_class_0_drifted[:, 0], X_class_0_drifted[:, 1], color='red', label='Class 0 (Drifted)', alpha=0.5)
    plt.scatter(X_class_1_drifted[:, 0], X_class_1_drifted[:, 1], color='orange', label='Class 1 (Drifted)', alpha=0.5)
    plt.scatter(X_class_2_drifted[:, 0], X_class_2_drifted[:, 1], color='pink', label='Class 3 (Drifted)', alpha=0.5)
    plt.title("Original and Drifted Gaussian Datasets for Classification")
    plt.legend()
    plt.show()

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    # Compute measures
    measures = DriftDatasetsCharacteristics.compute_dataset_measures_multiclass(X_original, X_drifted, y_original, y_drifted,
                                                                     0, drift_type, drift_location)

    return X_combined, y_combined, measures


if __name__=="__main__":
    X,y, measures = get_DS17()
    # Print results
    for key, value in measures.items():
        print(f"{key}: {value}")