import os
import numpy as np

def calculate_no_of_base_model_points(no_of_data_points, base_model_percent=10):
    """
    compute the number of base model points
    which is usually a percent of the total points like 1%, 10%
    """
    calculate_start_points = int(no_of_data_points * base_model_percent / 100)
    return calculate_start_points


def get_dataset_path(file_name):
    """
    get the dataset path stored in the project directory.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'PublicDatasets', file_name)
    return path



def combine_two_datasets(xs, ys, xs_new, ys_new):
    """
    Combines two data sets, used to generate adversarial scenarios
    """
    temp1 = list(zip(xs, ys))
    temp2 = list(zip(xs_new, ys_new))
    temp = temp1 + temp2
    res1, res2 = zip(*temp)
    xn, yn = list(res1), list(res2)
    xs_new = np.array(xn)
    ys_new = np.array(yn, dtype=int)  # or float if your labels are continuous values
    return xs_new, ys_new


def shuffle_dataset(X, y):
    # Shuffle original dataset
    indices_original = np.arange(X.shape[0])
    np.random.shuffle(indices_original)
    X = X[indices_original]
    y = y[indices_original]
    return X, y


def get_path_to_save_generated_dataset_file(directory):
    """
    returns the needed path to save the generated figure.
    """
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    parent_folder_path = os.path.dirname(current_script_path)
    path = os.path.join(parent_folder_path, 'Datasets', 'Datasets_Generators_CSV', directory)
    return path


def create_directory(path):
    """
    creates directory of the specified path if not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder created at {path}")
    else:
        print(f"Folder already exists at {path}")

# from scipy.sparse import csr_matrix
# def handle_sparse_model(linear_model):
#     # Check if the model is a sparse matrix (Compressed Sparse Row format)
#     if isinstance(linear_model, np.ndarray) and isinstance(linear_model[0], csr_matrix):
#         # If it's a sparse matrix, convert it to a dense array (if needed)
#         print("Sparse matrix detected.")
#         dense_model = linear_model[0].toarray()  # Converting the sparse matrix to a dense numpy array
#         print(f"Dense model after conversion: {dense_model}")
#         return dense_model
#     elif isinstance(linear_model, np.float64):
#         # If it's a dense scalar, return as is
#         print("Dense scalar detected.")
#         return linear_model
#     else:
#         # Handle any unexpected types
#         print("Unexpected model type detected.")
#         return None