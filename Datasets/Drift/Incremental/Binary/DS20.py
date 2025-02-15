import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

from Utils import Util, DriftDatasetsCharacteristics

'''
DS20
Drift Type: Incremental
Total n_samples: 21K
Dimensions: 20
Drift Location: every 1500 from 4500 to 16500 => [4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000, 16500]
'''

def get_DS20():
    np.random.seed(42)
    n_samples_per_class = 750
    n_dimensions = 20
    drift_type = 'Incremental'
    drift_locations = [4500, 6000, 7500, 9000, 10500, 12000, 13500, 15000, 16500]
    X1, y1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)
    X1_1, y1_1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)
    X1_2, y1_2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0)

    X_c1 = np.vstack((X1, X1_1, X1_2))


    X2, y2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=3.0, mean_class_1=6.0)

    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_c1, X2)

    X3, y3 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=6.0, mean_class_1=9.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X2, X3)
    X4, y4 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=9.0, mean_class_1=12.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X3, X4)
    X5, y5 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=12.0, mean_class_1=15.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X4, X5)
    X6, y6 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=15.0, mean_class_1=18.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X5, X6)
    X7, y7 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=18.0, mean_class_1=21.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X6, X7)
    X8, y8 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=21.0, mean_class_1=24.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X7, X8)
    X9, y9 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=24.0, mean_class_1=27.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X8, X9)

    X10, y10 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=27.0, mean_class_1=30.0)
    X10_1, y10_1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=27.0, mean_class_1=30.0)
    X10_2, y10_2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=27.0, mean_class_1=30.0)

    X_final = np.vstack((X10,X10_1,X10_2))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X9, X_final)
    ###################
    # Combine class 0 and class 1 datasets
    X_original_for_measures = np.vstack((X1, X1_1, X1_2))
    y_original_for_measures = np.hstack((y1, y1_1, y1_2))
    X_drifted_for_measures = np.vstack((X10, X10_1, X10_2))
    y_drifted_for_measures = np.hstack((y10, y10_1, y10_2))
    # Compute measures
    measures = DriftDatasetsCharacteristics.compute_dataset_measures_binary(X_original=X_original_for_measures,
                                                                            X_drifted=X_drifted_for_measures,
                                                                            y_original=y_original_for_measures,
                                                                            y_drifted=y_drifted_for_measures,
                                                                            no_of_middle_points=12000,
                                                                            drift_type=drift_type,
                                                                            drift_location=drift_locations)
    DriftDatasetsCharacteristics.print_measures(measures)

    X_combined = np.vstack((X1, X1_1, X1_2, X2,X3,X4,X5,X6,X7,X8,X9,X10, X10_1, X10_2))
    y_combined = np.hstack((y1, y1_1, y1_2,y2,y3,y4,y5,y6,y7,y8,y9,y10, y10_1, y10_2))

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    return X_combined, y_combined


def get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0, mean_class_1):

    mean_class_0 = np.array([mean_class_0] * n_dimensions)  # Mean of class 0
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

if __name__=="__main__":
    X,y = get_DS20()
