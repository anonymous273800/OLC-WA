import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Utils import Util
from Utils import DriftDatasetsCharacteristics

'''
DS18
Drift Type: Abrupt - multiclass
Total n_samples: 30000 (3 classes each 5000 + 3 drifted classes each 5000)
Dimensions: 20
Drift Location: Point 7500
'''

def get_DS18():
    np.random.seed(42)
    n_samples_per_class = 2500
    n_dimensions = 20
    drift_location = 7500
    drift_type = 'Abrupt'

    mean_class_0 = np.array([0.0] * n_dimensions)  # Mean vector for class 0 (10D)
    mean_class_1 = np.array([2.5] * n_dimensions)  # Mean vector for class 1 (10D)
    mean_class_2 = np.array([4.5] * n_dimensions)  # Mean vector for class 1 (10D)
    # Covariance matrix for both classes (10D)
    covariance = np.eye(n_dimensions)  # Identity matrix (10x10), implying independent features
    # the diagonal elements means the variance of each feature from the mean is 1, the off-diagonal (0's)
    # mean no covariance (correlation) between the features
    '''        
    1.0 & 0 & 0 & ... & 0 
    0 & 1.0 & 0 & ... & 0
    0 & 0 & 1.0 & ... & 0
    ...
    0 & 0 & 0 & ... & 1.0
    '''

    # Generate original dataset for both classes
    X_class_0 = np.random.multivariate_normal(mean_class_0, covariance, n_samples_per_class)
    X_class_1 = np.random.multivariate_normal(mean_class_1, covariance, n_samples_per_class)
    X_class_2 = np.random.multivariate_normal(mean_class_2, covariance, n_samples_per_class)

    # Labels for each class
    y_class_0 = np.zeros(n_samples_per_class)  # Class 0 label
    y_class_1 = np.ones(n_samples_per_class)  # Class 1 label
    y_class_2 = np.full(n_samples_per_class,2)  # Class 1 label

    # Combine class 0 and class 1 datasets
    X_original = np.vstack((X_class_0, X_class_1, X_class_2))
    y_original = np.hstack((y_class_0, y_class_1, y_class_2))

    # shuffle original dataset
    X_original, y_original = Util.shuffle_dataset(X_original, y_original)

    # Parameters for the drifted dataset (shift the mean of one or both classes)
    mean_class_0_drifted = np.array([10.0] * n_dimensions)  # Abrupt new mean for class 0 after drift
    mean_class_1_drifted = np.array([14.0] * n_dimensions)  # Abrupt new mean for class 1 after drift
    mean_class_2_drifted = np.array([18.0] * n_dimensions)  # Abrupt new mean for class 2 after drift

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

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X_combined = scaler.fit_transform(X_combined)

    measures = DriftDatasetsCharacteristics.compute_dataset_measures_multiclass(X_original, X_drifted, y_original,
                                                                                y_drifted, 0, drift_type,
                                                                                drift_location)
    return X_combined, y_combined, measures



if __name__=="__main__":
    X,y, measures = get_DS18()
    # Print results
    for key, value in measures.items():
        print(f"{key}: {value}")