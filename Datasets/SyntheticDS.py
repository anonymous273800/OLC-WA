from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def create_dataset(n_samples, n_features, n_classes, n_clusters_per_class=1,
                   n_redundant=0, n_informative=2, flip_y=0.0, class_sep=1.0, shuffle=True,
                   random_state=42):
    X, y = datasets.make_classification(n_samples=n_samples,  # Number of samples
                                        n_features=n_features,  # Number of features
                                        n_classes=n_classes,  # Number of classes
                                        n_clusters_per_class=n_clusters_per_class,  # Number of clusters per class
                                        # (n_classes * n_clusters_per_class must be smaller or equal 2**n_informative)
                                        n_redundant=n_redundant,  # Introduces features that are linear combinations
                                        # of the informative features, simulating real-world datasets where features
                                        # are often correlated and not all of them are independent.
                                        n_informative=n_informative,  # Number of informative features responsible
                                        # for determining the class labels.
                                        flip_y=flip_y,  # Fraction of samples with flipped labels (noise).
                                        class_sep=class_sep,  # Higher values increase the separation between classes.
                                        # Common values include 0, 1 (default), 2, 3.
                                        shuffle=shuffle,  # If True, the samples are shuffled randomly.
                                        random_state=random_state  # For reproducibility, ensures the same random
                                        # numbers are generated each time the code is run.
                                        )

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

