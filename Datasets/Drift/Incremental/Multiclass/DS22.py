import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

from Utils import Util, DriftDatasetsCharacteristics

'''
DS17
Drift Type: Incremental
Total n_samples: 42000
Dimensions: 20
Drift Location: every 3000 point starting 9000 to ... [9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000]
'''

def get_DS22():
    np.random.seed(42)
    n_samples_per_class = 1000
    n_dimensions = 20
    drift_type = 'Incremental'
    drift_locations = [9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 33000]
    X1, y1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0, mean_class_2=6.0)
    X1_1, y1_1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0, mean_class_2=6.0)
    X1_2, y1_2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=0.0, mean_class_1=3.0, mean_class_2=6.0)

    X_start = np.vstack((X1, X1_1, X1_2))

    X2, y2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=6.0, mean_class_1=9.0, mean_class_2=12.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X_start, X2)
    X3, y3 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=12.0, mean_class_1=15.0, mean_class_2=18.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X2, X3)
    X4, y4 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=18.0, mean_class_1=21.0, mean_class_2=24.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X3, X4)
    X5, y5 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=24.0, mean_class_1=27.0, mean_class_2=30.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X4, X5)
    X6, y6 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=30.0, mean_class_1=33.0, mean_class_2=36.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X5, X6)
    X7, y7 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=36.0, mean_class_1=39.0, mean_class_2=42.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X6, X7)
    X8, y8 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=42.0, mean_class_1=45.0, mean_class_2=48.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X7, X8)
    X9, y9 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=48.0, mean_class_1=51.0, mean_class_2=54.0)
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X8, X9)

    X10, y10 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=54.0, mean_class_1=57.0, mean_class_2=60.0)
    X10_1, y10_1 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=54.0, mean_class_1=57.0,mean_class_2=60.0)
    X10_2, y10_2 = get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0=54.0, mean_class_1=57.0,mean_class_2=60.0)

    X_end = np.vstack((X10, X10_1, X10_2))
    DriftDatasetsCharacteristics.print_drift_magnitude_consecutive(X9, X_end)

    ###################
    # Combine class 0 and class 1 datasets
    X_original_for_measures = np.vstack((X1, X1_1, X1_2))
    y_original_for_measures = np.hstack((y1, y1_1, y1_2))
    X_drifted_for_measures = np.vstack((X10, X10_1, X10_2))
    y_drifted_for_measures = np.hstack((y10, y10_1, y10_2))
    # Compute measures
    measures = DriftDatasetsCharacteristics.compute_dataset_measures_multiclass(X_original=X_original_for_measures,
                                                                                X_drifted=X_drifted_for_measures,
                                                                                y_original=y_original_for_measures,
                                                                                y_drifted=y_drifted_for_measures,
                                                                                no_of_middle_points=24000,
                                                                                drift_type=drift_type,
                                                                                drift_location=drift_locations)
    DriftDatasetsCharacteristics.print_measures(measures)
    ##########################################

    X_combined = np.vstack((X1, X1_1, X1_2, X2,X3,X4,X5,X6,X7,X8,X9,X10, X10_1, X10_2))
    y_combined = np.hstack((y1, y1_1, y1_2,y2,y3,y4,y5,y6,y7,y8,y9,y10,y10_1, y10_2))

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    return X_combined, y_combined


def get_mini_dataset(n_samples_per_class, n_dimensions, mean_class_0, mean_class_1, mean_class_2):

    mean_class_0 = np.array([mean_class_0] * n_dimensions)  # Mean of class 0
    mean_class_1 = np.array([mean_class_1] * n_dimensions)  # Mean of class 1
    mean_class_2 = np.array([mean_class_2] * n_dimensions)

    # mean_class_2 = (mean_class_0 + mean_class_1) / 3 + 2.0  # Adjust class 2 closer
    # mean_class_2 = np.array([mean_class_2] * n_dimensions)  # Mean of class 1

    covariance = np.eye(n_dimensions)

    # Generate original dataset for both classes
    X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
    X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)
    X_class_2 = np.random.multivariate_normal(mean_class_2, covariance, n_samples_per_class)

    # Labels for each class
    y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
    y_class_1 = np.ones(n_samples_per_class)  # Class 1 label
    y_class_2 = np.full(n_samples_per_class,2)  # Class 1 label

    # Combine class 0 and class 1 datasets
    X = np.vstack((X_class_0, X_class_1, X_class_2))
    y = np.hstack((y_class_0, y_class_1, y_class_2))

    # shuffle original dataset
    X, y = Util.shuffle_dataset(X, y)

    return X, y

if __name__=="__main__":
    X,y = get_DS22()
    print(X.shape)
