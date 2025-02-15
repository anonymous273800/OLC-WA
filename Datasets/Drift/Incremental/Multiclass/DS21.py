import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Utils import Util, DriftDatasetsCharacteristics

warnings.filterwarnings("ignore")
'''
DS21
Drift Type: Incremental
Total n_samples: 4200
Dimensions: 2
Classes 3
Drift Location: at 900, and every 300 point [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]
'''







# def get_DS21():
#     np.random.seed(42)
#     n_samples_per_class = 100
#     drift_type = "Incremental"
#     drift_locations = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]
#     X1, y1 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[0, 0, 0],
#         mean_class_1=[3, 3, 3],
#         mean_class_2_z_offset=2.0
#     )
#     X1_1, y1_1 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[0, 0, 0],
#         mean_class_1=[3, 3, 3],
#         mean_class_2_z_offset=2.0
#     )
#     X1_2, y1_2 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[0, 0, 0],
#         mean_class_1=[3, 3, 3],
#         mean_class_2_z_offset=2.0
#     )
#
#     X_start = np.vstack((X1, X1_1, X1_2))
#
#     X2, y2 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[3, 3, 3],
#         mean_class_1=[6, 6, 6],
#         mean_class_2_z_offset=2.0
#     )
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_start, X2)
#
#     X3, y3 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[6, 6, 6],
#         mean_class_1=[9, 9, 9],
#         mean_class_2_z_offset=2.0
#     )
#
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X2, X3)
#
#
#     X4, y4 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[9, 9, 9],
#         mean_class_1=[12, 12, 12],
#         mean_class_2_z_offset=2.0
#     )
#
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X3, X4)
#
#     X5, y5 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[12, 12, 12],
#         mean_class_1=[15, 15, 15],
#         mean_class_2_z_offset=2.0
#     )
#
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X4, X5)
#
#     X6, y6 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[15, 15, 15],
#         mean_class_1=[18, 18, 18],
#         mean_class_2_z_offset=2.0
#     )
#
#
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X5, X6)
#
#     X7, y7 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[18, 18, 18],
#         mean_class_1=[21, 21, 21],
#         mean_class_2_z_offset=2.0
#     )
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X6, X7)
#
#     X8, y8 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[21, 21, 21],
#         mean_class_1=[24, 24, 24],
#         mean_class_2_z_offset=2.0
#     )
#
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X7, X8)
#
#     X9, y9 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[21, 21, 21],
#         mean_class_1=[24, 24, 24],
#         mean_class_2_z_offset=2.0
#     )
#
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X8, X9)
#
#     X10, y10 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[24, 24, 24],
#         mean_class_1=[28, 28, 28],
#         mean_class_2_z_offset=2.0
#     )
#     X10_1, y10_1 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[24, 24, 24],
#         mean_class_1=[28, 28, 28],
#         mean_class_2_z_offset=2.0
#     )
#     X10_2, y10_2 = get_mini_dataset(
#         n_samples_per_class=n_samples_per_class,
#         mean_class_0=[24, 24, 24],
#         mean_class_1=[28, 28, 28],
#         mean_class_2_z_offset=2.0
#     )
#     X_end = np.vstack((X10, X10_1, X10_2))
#     DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X9, X_end)
#     ###################
#     # Combine class 0 and class 1 datasets
#     X_original_for_measures = np.vstack((X1, X1_1, X1_2))
#     y_original_for_measures = np.hstack((y1, y1_1, y1_2))
#     X_drifted_for_measures = np.vstack((X10, X10_1, X10_2))
#     y_drifted_for_measures = np.hstack((y10, y10_1, y10_2))
#     # Compute measures
#     measures = DriftDatasetsCharacteristics.compute_dataset_measures_multiclass(X_original=X_original_for_measures,
#                                                                             X_drifted=X_drifted_for_measures,
#                                                                             y_original=y_original_for_measures,
#                                                                             y_drifted=y_drifted_for_measures,
#                                                                             no_of_middle_points=2400,
#                                                                             drift_type=drift_type,
#                                                                             drift_location=drift_locations)
#     DriftDatasetsCharacteristics.print_measures(measures)
#     ##########################################
#
#     X_combined = np.vstack((X1,X1_1, X1_2, X2,X3,X4,X5,X6,X7,X8,X9,X10, X10_1, X10_2))
#     y_combined = np.hstack((y1, y1_1, y1_2,y2,y3,y4,y5,y6,y7,y8,y9,y10, y10_1, y10_2))
#
#     # X_combined = np.vstack((X1, X1_1, X1_2, X2))
#     # y_combined = np.hstack((y1, y1_1, y1_2, y2))
#
#     # Normalize the features using StandardScaler
#     scaler = StandardScaler()
#     X_combined = scaler.fit_transform(X_combined)
#
#     return X_combined, y_combined
#
# def get_mini_dataset(n_samples_per_class, mean_class_0, mean_class_1, mean_class_2_z_offset=1.0):
#     """
#     Generate a 3D dataset with three classes.
#
#     Args:
#         n_samples_per_class (int): Number of samples per class.
#         mean_class_0 (list or np.array): Mean for class 0 (3D).
#         mean_class_1 (list or np.array): Mean for class 1 (3D).
#         mean_class_2_z_offset (float): Z offset for class 2's mean.
#
#     Returns:
#         X (np.array): Feature matrix of shape (n_samples, 3).
#         y (np.array): Labels of shape (n_samples,).
#     """
#     # Compute means for the three classes in 3D
#     mean_class_0 = np.array(mean_class_0)  # Mean of class 0
#     mean_class_1 = np.array(mean_class_1)  # Mean of class 1
#
#     # Calculate mean for class 2 with offset adjustments
#     mean_class_2 = (mean_class_0 + mean_class_1) / 2
#     mean_class_2[2] += mean_class_2_z_offset  # Offset in z-dimension
#
#     covariance = np.array([[1.0, 0.5, 0.2],
#                            [0.5, 1.0, 0.3],
#                            [0.2, 0.3, 1.0]])  # Covariance matrix
#
#     # Generate datasets for each class
#     X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
#     X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)
#     X_class_2 = np.random.multivariate_normal(mean_class_2, covariance, n_samples_per_class)
#
#     # Labels for each class
#     y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
#     y_class_1 = np.ones(n_samples_per_class)  # Class 1 label
#     y_class_2 = np.full(n_samples_per_class, 2)  # Class 2 label
#
#     plotting_enabled = False
#     if plotting_enabled:
#         fig = plt.figure()
#         ax = fig.add_subplot(111, projection='3d')
#         ax.scatter(X_class_0[:, 0], X_class_0[:, 1], X_class_0[:, 2], color='red', label='Class 0', alpha=0.5)
#         ax.scatter(X_class_1[:, 0], X_class_1[:, 1], X_class_1[:, 2], color='orange', label='Class 1', alpha=0.5)
#         ax.scatter(X_class_2[:, 0], X_class_2[:, 1], X_class_2[:, 2], color='purple', label='Class 2', alpha=0.5)
#         plt.title("3D Gaussian Datasets for Classification")
#         plt.legend()
#         plt.show()
#
#     # Combine datasets
#     X = np.vstack((X_class_0, X_class_1, X_class_2))
#     y = np.hstack((y_class_0, y_class_1, y_class_2))
#
#     # Shuffle dataset
#     rng = np.random.default_rng()
#     indices = rng.permutation(X.shape[0])
#     X = X[indices]
#     y = y[indices]
#
#     return X, y



def get_DS21():
    np.random.seed(42)
    n_samples_per_class = 100
    drift_type = "Incremental"
    drift_locations = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300]

    def create_mini_dataset_2d(mean_class_0, mean_class_1):
        return get_mini_dataset(
            n_samples_per_class=n_samples_per_class,
            mean_class_0=mean_class_0,
            mean_class_1=mean_class_1
        )

    # Generate datasets
    X1, y1 = create_mini_dataset_2d([0, 0], [3, 3])
    X1_1, y1_1 = create_mini_dataset_2d([0, 0], [3, 3])
    X1_2, y1_2 = create_mini_dataset_2d([0, 0], [3, 3])
    X_start = np.vstack((X1, X1_1, X1_2))

    X2, y2 = create_mini_dataset_2d([3, 3], [6, 6])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_start, X2)

    X3, y3 = create_mini_dataset_2d([6, 6], [9, 9])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X2, X3)

    X4, y4 = create_mini_dataset_2d([9, 9], [12, 12])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X3, X4)

    X5, y5 = create_mini_dataset_2d([12, 12], [15, 15])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X4, X5)

    X6, y6 = create_mini_dataset_2d([15, 15], [18, 18])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X5, X6)

    X7, y7 = create_mini_dataset_2d([18, 18], [21, 21])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X6, X7)

    X8, y8 = create_mini_dataset_2d([21, 21], [24, 24])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X7, X8)

    X9, y9 = create_mini_dataset_2d([24, 24], [28, 28])
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X8, X9)

    X10, y10 = create_mini_dataset_2d([28, 28], [32, 32])
    X10_1, y10_1 = create_mini_dataset_2d([28, 28], [32, 32])
    X10_2, y10_2 = create_mini_dataset_2d([28, 28], [32, 32])
    X_end = np.vstack((X10, X10_1, X10_2))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X9, X_end)

    # Combine class 0 and class 1 datasets for measures
    X_original_for_measures = np.vstack((X1, X1_1, X1_2))
    y_original_for_measures = np.hstack((y1, y1_1, y1_2))
    X_drifted_for_measures = np.vstack((X10, X10_1, X10_2))
    y_drifted_for_measures = np.hstack((y10, y10_1, y10_2))

    # Compute measures
    measures = DriftDatasetsCharacteristics.compute_dataset_measures_multiclass(
        X_original=X_original_for_measures,
        X_drifted=X_drifted_for_measures,
        y_original=y_original_for_measures,
        y_drifted=y_drifted_for_measures,
        no_of_middle_points=2400,
        drift_type=drift_type,
        drift_location=drift_locations
    )
    DriftDatasetsCharacteristics.print_measures(measures)

    # Combine all datasets
    X_combined = np.vstack((X1, X1_1, X1_2, X2, X3, X4, X5, X6, X7, X8, X9, X10, X10_1, X10_2))
    y_combined = np.hstack((y1, y1_1, y1_2, y2, y3, y4, y5, y6, y7, y8, y9, y10, y10_1, y10_2))

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    return X_combined, y_combined



def get_mini_dataset(n_samples_per_class, mean_class_0, mean_class_1):
    """
    Generate a 2D dataset with three classes.

    Args:
        n_samples_per_class (int): Number of samples per class.
        mean_class_0 (list or np.array): Mean for class 0 (2D).
        mean_class_1 (list or np.array): Mean for class 1 (2D).

    Returns:
        X (np.array): Feature matrix of shape (n_samples, 2).
        y (np.array): Labels of shape (n_samples,).
    """
    # Compute means for the three classes in 2D
    mean_class_0 = np.array(mean_class_0)  # Mean of class 0
    mean_class_1 = np.array(mean_class_1)  # Mean of class 1

    # Calculate mean for class 2 with adjustments
    mean_class_2 = (mean_class_0 + mean_class_1) / 2  # Midpoint between class 0 and class 1

    covariance = np.array([[1.0, 0.5],
                           [0.5, 1.0]])  # Covariance matrix for 2D

    # Generate datasets for each class
    X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
    X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)
    X_class_2 = np.random.multivariate_normal(mean_class_2, covariance, n_samples_per_class)

    # Labels for each class
    y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
    y_class_1 = np.ones(n_samples_per_class)  # Class 1 label
    y_class_2 = np.full(n_samples_per_class, 2)  # Class 2 label

    plotting_enabled = False
    if plotting_enabled:
        plt.scatter(X_class_0[:, 0], X_class_0[:, 1], color='red', label='Class 0', alpha=0.5)
        plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='orange', label='Class 1', alpha=0.5)
        plt.scatter(X_class_2[:, 0], X_class_2[:, 1], color='purple', label='Class 2', alpha=0.5)
        plt.title("2D Gaussian Datasets for Classification")
        plt.legend()
        plt.show()

    # Combine datasets
    X = np.vstack((X_class_0, X_class_1, X_class_2))
    y = np.hstack((y_class_0, y_class_1, y_class_2))

    # Shuffle dataset
    rng = np.random.default_rng()
    indices = rng.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    return X, y


def plot_datasets(X, y):
    colors = plt.cm.get_cmap('tab10', 10)  # 10 distinct colors
    plt.figure(figsize=(10, 8))

    for i in range(10):
        # Plot every 100 samples with distinct color (each mini-dataset)
        X_mini = X[i*100:(i+1)*100]
        y_mini = y[i*100:(i+1)*100]
        plt.scatter(X_mini[y_mini == 0][:, 0], X_mini[y_mini == 0][:, 1], color=colors(i), label=f'Dataset {i+1} Class 0')
        plt.scatter(X_mini[y_mini == 1][:, 0], X_mini[y_mini == 1][:, 1], color=colors(i), marker='x', label=f'Dataset {i+1} Class 1')

    plt.title("10 Mini Datasets: Class 0 and Class 1")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

from sklearn.linear_model import LogisticRegression

if __name__=="__main__":
    X,y = get_DS21()
    plot_datasets(X,y)
    # X = X[:150]
    # y = y[:150]
    # model = LogisticRegression()
    # model.fit(X,y)


