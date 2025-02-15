import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

from Utils import Util, DriftDatasetsCharacteristics

'''
DS23
Drift Type: Gradual
Total n_samples: 2600
Dimensions: 2
Drift Location: 800 | 1000 | 1200 | 1600 | 1800 
'''



def get_DS23():
    np.random.seed(42)
    n_samples_per_class = 100
    drift_type = 'Gradual'
    drift_locations = [800 , 1000 , 1200 , 1600 , 1800 ]

    X1, y1 = get_mini_dataset(n_samples_per_class, mean_class_0=0.0, mean_class_1=2.0)
    X2, y2 = get_mini_dataset(n_samples_per_class, mean_class_0=0.0, mean_class_1=2.0)
    X3, y3 = get_mini_dataset(n_samples_per_class, mean_class_0=0.0, mean_class_1=2.0)
    X4, y4 = get_mini_dataset(n_samples_per_class, mean_class_0=0.0, mean_class_1=2.0)
    X_start = np.vstack((X1,X2,X3,X4))


    X5, y5 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_start, X5)

    X6, y6 = get_mini_dataset(n_samples_per_class, mean_class_0=0.0, mean_class_1=2.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X5, X6)

    X7, y7 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    X8, y8 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    X_middle = np.vstack((X7,X8))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X6, X_middle)

    X9, y9 = get_mini_dataset(n_samples_per_class, mean_class_0=0.0, mean_class_1=2.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_middle, X9)

    X10, y10 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    X11, y11 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    X12, y12 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    X13, y13 = get_mini_dataset(n_samples_per_class, mean_class_0=15.0, mean_class_1=17.0)
    X_end = np.vstack((X10, X11, X12, X13))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X9, X_end)

    ###################
    # Combine class 0 and class 1 datasets
    X_original_for_measures = np.vstack((X1, X2, X3,X4))
    y_original_for_measures = np.hstack((y1, y2, y3, y4))
    X_drifted_for_measures = np.vstack((X10, X11, X12, X13))
    y_drifted_for_measures = np.hstack((y10, y11, y12, y13))
    # Compute measures
    measures = DriftDatasetsCharacteristics.compute_dataset_measures_binary(X_original=X_original_for_measures, X_drifted=X_drifted_for_measures, y_original=y_original_for_measures,
                                                                            y_drifted=y_drifted_for_measures, no_of_middle_points=1000,drift_type=drift_type, drift_location=drift_locations)
    DriftDatasetsCharacteristics.print_measures(measures)

    ###############################

    X_combined = np.vstack((X1, X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12, X13))
    y_combined = np.hstack((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10, y11, y12, y13))

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    return X_combined, y_combined



def plot_datasets(X, y):
    colors = plt.cm.get_cmap('tab20', 12)  # 10 distinct colors
    plt.figure(figsize=(10, 8))

    for i in range(12):
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


def get_mini_dataset(n_samples_per_class, mean_class_0, mean_class_1):

    mean_class_0 = np.array([mean_class_0, mean_class_0])  # Mean of class 0
    mean_class_1 = np.array([mean_class_1, mean_class_1])  # Mean of class 1
    covariance = np.array([[1.0, 0.5], [0.5, 1.0]])  # Same covariance for both classes

    # Generate original dataset for both classes
    X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
    X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)

    # Labels for each class
    y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
    y_class_1 = np.ones(n_samples_per_class)  # Class 1 label

    plotting_enabled = False
    if plotting_enabled == True:
        plt.scatter(X_class_0[:, 0], X_class_0[:, 1], color='red', label='Class 0', alpha=0.5)
        plt.scatter(X_class_1[:, 0], X_class_1[:, 1], color='orange', label='Class 1', alpha=0.5)
        plt.title("Original and Drifted Gaussian Datasets for Classification")
        plt.legend()
        plt.show()

    # Combine class 0 and class 1 datasets
    X = np.vstack((X_class_0, X_class_1))
    y = np.hstack((y_class_0, y_class_1))

    # shuffle original dataset
    X, y = Util.shuffle_dataset(X, y)

    return X, y

if __name__=="__main__":
    X,y = get_DS23()
    plot_datasets(X,y)