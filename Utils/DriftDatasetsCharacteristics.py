
import numpy as np


def compute_dataset_measures_binary(X_original, X_drifted, y_original, y_drifted, no_of_middle_points,drift_type, drift_location):
    measures = {}

    # Total dataset size and dimensions
    measures["drift_type"] = drift_type
    measures["total_samples"] = X_original.shape[0] + X_drifted.shape[0] + no_of_middle_points
    measures["dimensions"] = X_original.shape[1]
    measures["classes"] = len(np.unique(y_original))
    measures["drift_location"] = drift_location

    # Class distributions
    unique, counts = np.unique(np.hstack((y_original, y_drifted)), return_counts=True)
    measures["class_distribution"] = dict(zip(unique.astype(int), counts +no_of_middle_points/2))

    # Concept Distance: Distance between the means of the two classes (original data)
    mean_class_0 = np.mean(X_original[y_original == 0], axis=0)
    mean_class_1 = np.mean(X_original[y_original == 1], axis=0)
    concept_distance = np.linalg.norm(mean_class_1 - mean_class_0)
    measures["original_classes_distance"] = concept_distance

    # Concept Distance: Distance between the means of the two classes (drifted data)
    mean_class_0_drifted = np.mean(X_drifted[y_drifted == 0], axis=0)
    mean_class_1_drifted = np.mean(X_drifted[y_drifted == 1], axis=0)
    concept_distance_drifted = np.linalg.norm(mean_class_1_drifted - mean_class_0_drifted)
    measures["drifted_classes_distance"] = concept_distance_drifted


    ######################################################
    measures["snr-original"] = compute_snr(X_original, y_original)
    measures["snr-drifted"] = compute_snr(X_drifted, y_drifted)

    ######################################################

    # Drift Magnitude: Distance between original and drifted means for both original and drifted datasets
    # we compute the mean of the original (for all classes) and the mean of the drifted (for all classes) and then subtract (or
    # in other words compute the euclidian distance for both means)



    measures["drift_magnitude_end_mean_and_start_mean_ecludian_distance"] = compute_drift_magnitude(X_original, X_drifted)

    return measures





def compute_snr(X, y):
    mean_class_0 = X[y == 0].mean(axis=0)
    mean_class_1 = X[y == 1].mean(axis=0)

    # Signal (Euclidean distance between means)
    signal = np.linalg.norm(mean_class_0 - mean_class_1)

    # Noise (average variance within each class)
    variance_class_0 = X[y == 0].var(axis=0).mean()
    variance_class_1 = X[y == 1].var(axis=0).mean()
    noise = (variance_class_0 + variance_class_1) / 2

    # Signal-to-Noise Ratio
    snr = signal / noise
    return snr



def compute_drift_magnitude(X_original, X_drifted):
    mean_class_original_dataset_all_classes = np.mean(X_original, axis=0)
    mean_class_drifted_dataset_all_classes = np.mean(X_drifted, axis=0)
    drift_magnitude_end_mean_and_start_mean_ecludian_distance = np.linalg.norm(
        mean_class_drifted_dataset_all_classes - mean_class_original_dataset_all_classes)

    return drift_magnitude_end_mean_and_start_mean_ecludian_distance


def compute_dataset_measures_multiclass(X_original, X_drifted, y_original, y_drifted, no_of_middle_points, drift_type, drift_location):
    measures = {}

    no_of_classes = len(np.unique(y_original))
    # Total dataset size and dimensions
    measures["total_samples"] = X_original.shape[0] + X_drifted.shape[0] + no_of_middle_points
    measures["dimensions"] = X_original.shape[1]
    measures["drift_location"] = drift_location
    measures["classes"] = no_of_classes

    # Class distributions
    unique, counts = np.unique(np.hstack((y_original, y_drifted)), return_counts=True)
    measures["class_distribution"] = dict(zip(unique.astype(int), counts + no_of_middle_points/no_of_classes))




    # Means for each class (original and drifted)
    means_original = {cls: np.mean(X_original[y_original == cls], axis=0) for cls in np.unique(y_original)}
    means_drifted = {cls: np.mean(X_drifted[y_drifted == cls], axis=0) for cls in np.unique(y_drifted)}

    # Concept Distance: Pairwise distances between class means (original data)
    concept_distances = {}
    for i in means_original:
        for j in means_original:
            if i < j:  # Avoid duplicate pair comparisons
                distance = np.linalg.norm(means_original[i] - means_original[j])
                concept_distances[f"concept_distance_{i}_{j}"] = distance

    measures["concept_distances"] = concept_distances

    # Drifted Concept Distance: Pairwise distances between class means (drifted data)
    drifted_concept_distances = {}
    for i in means_drifted:
        for j in means_drifted:
            if i < j:
                distance = np.linalg.norm(means_drifted[i] - means_drifted[j])
                drifted_concept_distances[f"drifted_concept_distance_{i}_{j}"] = distance

    measures["drifted_concept_distances"] = drifted_concept_distances

    # SNR computation for both original and drifted datasets
    measures["snr-original"] = compute_snr_multiclass(X_original, y_original)
    measures["snr-drifted"] = compute_snr_multiclass(X_drifted, y_drifted)

    # Drift Magnitude: Distance between original and drifted means for both original and drifted datasets
    measures["drift_magnitude_end_mean_and_start_mean_ecludian_distance"] = compute_drift_magnitude(X_original, X_drifted)

    return measures


def compute_snr_multiclass(X, y):
    # Means for each class
    means = {cls: np.mean(X[y == cls], axis=0) for cls in np.unique(y)}

    # Signal computation: Pairwise distances between class means
    signal = 0
    for i in means:
        for j in means:
            if i < j:
                signal += np.linalg.norm(means[i] - means[j])

    # Noise computation: Average variance across all classes
    noise = 0
    for cls in np.unique(y):
        noise += X[y == cls].var(axis=0).mean()
    noise /= len(np.unique(y))

    # SNR calculation
    snr = signal / noise
    return snr


def print_drift_magnitude_consecutive(X_c1, X_c2):
    dm = compute_drift_magnitude(X_c1, X_c2)
    print("D.M c1 - c2 (Consec.)", dm)


def print_measures(measures):
    for key, value in measures.items():
        print(f"{key}: {value}")
    print('---------------------------------------------------------')