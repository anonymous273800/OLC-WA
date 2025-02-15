import warnings
import Hyperparameters.Hyperparameters
import numpy as np
from Datasets import SyntheticDS
from Utils import Constants
import time
from Hyperparameters import Hyperparameters
from Models.BatchClassification import BatchClassification
from Models.OLCWA import OLCWA
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.WidrowHoff import WidrowHoff
from Hyperparameters import Hyperparameters
from Models.Perceptron import Perceptron
from Models.Online_Logistic_Regression import Online_Logistic_Regression
from Models.OnlineNaiveBayes import OnlineNaiveBayes
from Models.OnlineSupportVectorMachine import OnlineSupportVectorMachine
from Models.HoeffdingTree import HoeffdingTree
warnings.filterwarnings("ignore")

def experiment4Main():
    n_samples = 50000
    n_features = 500
    n_classes = 2
    n_clusters_per_class = 1
    n_redundant = 0
    n_informative = 2
    flip_y = 0.5 # 50%
    class_sep = 1
    shuffle = True

    SEEDS = Constants.SEEDS

    number_of_seeds = len(SEEDS)
    number_of_folds = 5
    total_runs = number_of_seeds * number_of_folds

    batch_acc_list_per_seed = []
    olc_wa_acc_list_per_seed = []
    olc_wa_sccm_acc_list_per_seed = []
    pa_acc_list_per_seed = []
    pa2_acc_list_per_seed = []
    widrow_hoff_acc_list_per_seed = []
    perceptron_acc_list_per_seed = []
    online_logistic_regression_acc_list_per_seed = []
    online_naive_bayes_acc_list_per_seed = []
    hoeffding_tree_acc_list_per_seed = []

    batch_execution_time = 0
    olc_wa_execution_time = 0
    olc_wa_sccm_execution_time = 0
    pa_execution_time = 0
    pa2_execution_time = 0
    widrow_hoff_execution_time = 0
    perceptron_execution_time = 0
    online_logistic_regression_execution_time = 0
    online_naive_bayes_execution_time = 0
    hoeffding_tree_execution_time = 0

    # 1. Experiments_Binary
    for seed in SEEDS:
        X, y = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_clusters_per_class=n_clusters_per_class,
                                          n_redundant=n_redundant, n_informative=n_informative, flip_y=flip_y, class_sep=class_sep, shuffle=shuffle,
                                          random_state=seed)

        # # 1. Batch Experiment
        # start_time = time.perf_counter()
        # batch_acc = BatchClassification.batch_logistic_regression_KFold(X, y, seed, shuffle=True)
        # end_time = time.perf_counter()
        # batch_execution_time += (end_time - start_time)
        # batch_acc_list_per_seed.append(batch_acc)

        # # # 2. OLC_WA Experiment
        # # start_time = time.perf_counter()
        # # olc_wa_acc = OLCWA.olc_wa_classification_KFold(X, y,
        # #                                                .15, #Hyperparameters.olc_wa_w_inc,
        # #                                                Hyperparameters.olr_wa_base_model_size1,
        # #                                                Hyperparameters.olr_wa_increment_size2(n_features,
        # #                                                                                     user_defined_val=10),
        # #                                                seed,
        # #                                                shuffle=True
        # #                                                )
        # # end_time = time.perf_counter()
        # # olc_wa_execution_time+=(end_time - start_time)
        # # olc_wa_acc_list_per_seed.append(olc_wa_acc)
        # #
        # 2. OLC_WA (SCCM) Experiment
        start_time = time.perf_counter()
        kpi = 'ACC'
        multiplier = 1.5
        olc_wa_sccm_acc = OLCWA.olc_wa_sccm_classification_KFold(X, y,
                                                                 .15,  # Hyperparameters.olc_wa_w_inc,
                                                                 Hyperparameters.olr_wa_base_model_size1,
                                                                 Hyperparameters.olr_wa_increment_size2(n_features,
                                                                                                        user_defined_val=10),
                                                                 seed,
                                                                 shuffle=True,
                                                                 kpi=kpi,
                                                                 multiplier=multiplier,
                                                                 expr='expr4'
                                                                 )
        end_time = time.perf_counter()
        olc_wa_sccm_execution_time += (end_time - start_time)
        olc_wa_sccm_acc_list_per_seed.append(olc_wa_sccm_acc)
        # #
        # # # # Passive-Aggressive Experiment
        # # # # C = 1.0# .001
        # # # C = 0.01
        # # # start_time = time.perf_counter()
        # # # pa_acc = OnlinePassiveAggressive.online_passive_aggressive_KFold(X, y, C, seed)
        # # # end_time = time.perf_counter()
        # # # pa_execution_time += (end_time - start_time)
        # # # pa_acc_list_per_seed.append(pa_acc)
        # # #
        # # Passive-Aggressive
        # C = 0.01  # 1.0
        # start_time = time.perf_counter()
        # pa2_acc = OnlinePassiveAggressive.online_passive_aggressive_sk_learn_KFold(X, y, C, seed)
        # end_time = time.perf_counter()
        # pa2_execution_time += (end_time - start_time)
        # pa2_acc_list_per_seed.append(pa2_acc)
        # #
        # # 4. WidrowHoff
        # learning_rate = 0.001
        # start_time = time.perf_counter()
        # widrow_hoff_acc = WidrowHoff.widrow_hoff_KFold(X, y, learning_rate, seed)
        # end_time = time.perf_counter()
        # widrow_hoff_execution_time += (end_time - start_time)
        # widrow_hoff_acc_list_per_seed.append(widrow_hoff_acc)
        #
        # # 5. Perceptron
        # start_time = time.perf_counter()
        # perceptron_acc = Perceptron.perceptron_KFold(X, y, seed)
        # end_time = time.perf_counter()
        # perceptron_execution_time += (end_time - start_time)
        # perceptron_acc_list_per_seed.append(perceptron_acc)
        #
        # # 6. Online Logistic Regression
        # learning_rate = 0.0001
        # start_time = time.perf_counter()
        # online_logistic_regression_acc = Online_Logistic_Regression.online_logistic_regression_KFold(X, y, learning_rate, seed)
        # end_time = time.perf_counter()
        # online_logistic_regression_execution_time += (end_time - start_time)
        # online_logistic_regression_acc_list_per_seed.append(online_logistic_regression_acc)

        # 7. Online Naive Bayes
        start_time = time.perf_counter()
        online_naive_bayes_acc = OnlineNaiveBayes.online_naive_bayes_KFold(X, y, seed)
        end_time = time.perf_counter()
        online_naive_bayes_execution_time += (end_time - start_time)
        online_naive_bayes_acc_list_per_seed.append(online_naive_bayes_acc)
        #
        # # 8. Hoeffding Tree
        # start_time = time.perf_counter()
        # hoeffding_tree_acc = HoeffdingTree.hoeffding_tree_KFold(X, y, seed)
        # end_time = time.perf_counter()
        # hoeffding_tree_execution_time += (end_time - start_time)
        # hoeffding_tree_acc_list_per_seed.append(hoeffding_tree_acc)

    # 1. Results for Batch Experiment:
    batch_acc = np.array(batch_acc_list_per_seed).mean()
    print('Batch (Logistic Regression), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(batch_execution_time / total_runs), ', accuracy:', "{:.5f}".format(batch_acc))

    # 2. Results for OLC_WA Experiment
    olc_wa_acc = np.array(olc_wa_acc_list_per_seed).mean()
    print('OLC_WA, 5 folds, seeds averaging. time:', "{:.5f} s".format(olc_wa_execution_time / total_runs), ', accuracy:',
          "{:.5f}".format(olc_wa_acc))

    # 3. Results for OLC_WA (SCCM) Experiment
    olc_wa_sccm_acc = np.array(olc_wa_sccm_acc_list_per_seed).mean()
    print('OLC_WA (SCCM), 5 folds, seeds averaging. time:', "{:.5f} s".format(olc_wa_sccm_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(olc_wa_sccm_acc))


    # 3. Results for PA Experiment
    pa_acc = np.array(pa_acc_list_per_seed).mean()
    print('PA, 5 folds, seeds averaging. time:', "{:.5f} s".format(pa_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(pa_acc))

    # 4. Results for PA Experiment
    pa2_acc = np.array(pa2_acc_list_per_seed).mean()
    print('PA2, 5 folds, seeds averaging. time:', "{:.5f} s".format(pa2_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(pa2_acc))

    # 3. Results for WidrowHoff Experiment
    widrow_hoff_acc = np.array(widrow_hoff_acc_list_per_seed).mean()
    print('WidrowHoff, 5 folds, seeds averaging. time:', "{:.5f} s".format(widrow_hoff_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(widrow_hoff_acc))

    # 4. Results for Perceptron Experiment
    perceptron_acc = np.array(perceptron_acc_list_per_seed).mean()
    print('Perceptron, 5 folds, seeds averaging. time:', "{:.5f} s".format(perceptron_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(perceptron_acc))

    # 5. Results for Online Logistic regression Experiment
    online_logistic_regression_acc = np.array(online_logistic_regression_acc_list_per_seed).mean()
    print('Online Logistic Regression, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_logistic_regression_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(online_logistic_regression_acc))

    # 6. Results for Online Naive Bayes Experiment
    online_naive_bayes_acc = np.array(online_naive_bayes_acc_list_per_seed).mean()
    print('Online Naive Bayes, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_naive_bayes_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(online_naive_bayes_acc))

    # 7. Results for Online Naive Bayes Experiment
    hoeffding_tree_acc = np.array(hoeffding_tree_acc_list_per_seed).mean()
    print('Hoeffding Tree, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(hoeffding_tree_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(hoeffding_tree_acc))


if __name__ == '__main__':
    experiment4Main()