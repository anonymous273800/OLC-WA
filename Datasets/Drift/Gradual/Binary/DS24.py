import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

from Utils import Util, DriftDatasetsCharacteristics

'''
DS24
Drift Type: Gradual
Total n_samples: 
Dimensions: 20
Drift Location: 8000 | 10000 | 12000 | 16000 | 18000
Drift Location: 
'''

def get_DS24():
    np.random.seed(42)
    n_samples_per_class = 1000
    n_dimensions = 20
    drift_type = 'Gradual'
    drift_locations = [8000 , 10000 , 12000 , 16000 , 18000]

    X1, y1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)
    X2, y2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)
    X3, y3 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)

    X_start = np.vstack((X1, X2, X3))

    X5, y5 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_start, X5)

    X6, y6 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X5, X6)

    X7, y7 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    X8, y8 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    X_middle = np.vstack((X7, X8))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X6, X_middle)

    X9, y9 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_middle, X9)

    X10, y10 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    X11, y11 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    X12, y12 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    X_end = np.vstack((X10, X11, X12,))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X9, X_end)
    ###################
    # Combine class 0 and class 1 datasets
    X_original_for_measures = np.vstack((X1, X2, X3))
    y_original_for_measures = np.hstack((y1, y2, y3))
    X_drifted_for_measures = np.vstack((X10, X11, X12))
    y_drifted_for_measures = np.hstack((y10, y11, y12))
    # Compute measures
    measures = DriftDatasetsCharacteristics.compute_dataset_measures_binary(X_original=X_original_for_measures,
                                                                            X_drifted=X_drifted_for_measures,
                                                                            y_original=y_original_for_measures,
                                                                            y_drifted=y_drifted_for_measures,
                                                                            no_of_middle_points=10000,
                                                                            drift_type=drift_type,
                                                                            drift_location=drift_locations)
    DriftDatasetsCharacteristics.print_measures(measures)

    ###############################

    X_combined = np.vstack((X1, X2,X3,X5,X6,X7,X8,X9,X10,X11,X12))
    y_combined = np.hstack((y1,y2,y3,y5,y6,y7,y8,y9,y10, y11, y12))

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    return X_combined, y_combined


def get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0, mean_class_1):

    mean_class_0 = np.array([mean_class_0]* n_dimensions)  # Mean of class 0
    mean_class_1 = np.array([mean_class_1] * n_dimensions)  # Mean of class 1
    covariance = np.eye(n_dimensions)

    # Generate original dataset for both classes
    X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
    X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)

    # Labels for each class
    y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
    y_class_1 = np.ones(n_samples_per_class)  # Class 1 label

    # Combine class 0 and class 1 datasets
    X = np.vstack((X_class_0, X_class_1))
    y = np.hstack((y_class_0, y_class_1))

    # shuffle original dataset
    X, y = Util.shuffle_dataset(X, y)

    return X, y


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

if __name__=="__main__":
    X,y = get_DS24()
    print(X.shape)
    plot_datasets(X,y)