from Datasets import SyntheticDS
from Utils import Constants, Util
import pandas as pd

def DS02():
    path_to_save_data_set = Util.get_path_to_save_generated_dataset_file('02_DS2')
    Util.create_directory(path_to_save_data_set)

    n_samples = 10000
    n_features = 20
    n_classes = 2
    n_clusters_per_class = 1
    n_redundant = 0
    n_informative = 2
    flip_y = 0.2  # 20%
    class_sep = 1
    shuffle = True
    SEEDS = Constants.SEEDS

    for seed in SEEDS:
        X, y = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                          n_clusters_per_class=n_clusters_per_class,
                                          n_redundant=n_redundant, n_informative=n_informative, flip_y=flip_y,
                                          class_sep=class_sep, shuffle=shuffle,
                                          random_state=seed)
        # Create a DataFrame using X and y
        df = pd.DataFrame(data=X, columns=[f"feature_{i}" for i in range(n_features)])
        df['target'] = y

        # Save DataFrame to a CSV file
        df.to_csv(path_to_save_data_set + '\\002_DS2_' + str(seed) + '.csv', index=False)


if __name__ == '__main__':
    DS02()