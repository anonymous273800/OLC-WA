import warnings
import Hyperparameters.Hyperparameters
import numpy as np
from Datasets import SyntheticDS
from Utils import Constants
import time
from Models.BatchClassification import BatchClassification
from Models.BatchClassificationMulticlass import BatchClassificationMulticlass
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.OLCWA import OLCWA, OLCWA_MC_ONE_VS_REST, OLCWA_MC_ONE_VS_ONE
from Models.Perceptron import Perceptron
from Models.WidrowHoff import WidrowHoff
from Models.Online_Logistic_Regression import Online_Logistic_Regression
from Models.OnlineNaiveBayes import OnlineNaiveBayes
from Models.HoeffdingTree import HoeffdingTree

from Hyperparameters import Hyperparameters
warnings.filterwarnings("ignore")

def experiment11Main():
    n_samples = 50000
    n_features = 500
    n_classes = 20
    n_clusters_per_class = 1
    n_redundant = 0
    n_informative = 500
    flip_y = 0.50 # 50%
    class_sep = 3
    shuffle = True


    SEEDS = Constants.SEEDS

    number_of_seeds = len(SEEDS)
    number_of_folds = 5
    total_runs = number_of_seeds * number_of_folds

    batch_acc_list_per_seed = []
    olc_wa_acc_list_per_seed = []
    olc_wa2_acc_list_per_seed = []
    olc_wa_sccm_acc_list_per_seed = []
    pa_acc_list_per_seed = []
    widrow_hoff_acc_list_per_seed = []
    widrow_hoff2_acc_list_per_seed = []
    perceptron_acc_list_per_seed = []
    perceptron2_acc_list_per_seed = []
    online_logistic_regression_acc_list_per_seed = []
    online_logistic_regression2_acc_list_per_seed = []
    online_naive_bayes_acc_list_per_seed = []
    hoeffding_tree_acc_list_per_seed = []

    batch_execution_time = 0
    olc_wa_execution_time = 0
    olc_wa2_execution_time = 0
    olc_wa_sccm_execution_time = 0
    pa_execution_time = 0
    widrow_hoff_execution_time = 0
    widrow_hoff2_execution_time = 0
    perceptron_execution_time = 0
    perceptron2_execution_time = 0
    online_logistic_regression_execution_time = 0
    online_logistic_regression2_execution_time = 0
    online_naive_bayes_execution_time = 0
    hoeffding_tree_execution_time = 0

    # 1. Experiments_Binary
    for seed in SEEDS:
        X, y = SyntheticDS.create_dataset(n_samples=n_samples, n_features=n_features, n_classes=n_classes,
                                          n_clusters_per_class=n_clusters_per_class,
                                          n_redundant=n_redundant, n_informative=n_informative, flip_y=flip_y,
                                          class_sep=class_sep, shuffle=shuffle,
                                          random_state=seed)

        # 1. Batch Experiment
        start_time = time.perf_counter()
        batch_acc = BatchClassificationMulticlass.batch_logistic_regression_multiclass_KFold(X, y, seed)
        end_time = time.perf_counter()
        batch_execution_time += (end_time - start_time)
        batch_acc_list_per_seed.append(batch_acc)
        #
        # # 2. OLC_WA Experiment (One-Vs-Rest)
        # start_time = time.perf_counter()
        # olc_wa_acc = OLCWA_MC_ONE_VS_REST.olc_wa_mc_classification_KFold(X, y,
        #                                                .3, #Hyperparameters.olc_wa_w_inc,
        #                                                Hyperparameters.olr_wa_base_model_size1,
        #                                                Hyperparameters.olr_wa_increment_size(n_features,
        #                                                                                    user_defined_val=10),
        #                                                seed
        #                                                )
        # end_time = time.perf_counter()
        # olc_wa_execution_time+=(end_time - start_time)
        # olc_wa_acc_list_per_seed.append(olc_wa_acc)

        # 2. OLC_WA (SCCM) Experiment (One-Vs-Rest)
        start_time = time.perf_counter()
        kpi = 'ACC'
        multiplier = 2.5
        olc_wa_sccm_acc = OLCWA_MC_ONE_VS_REST.olc_wa_sccm_mc_classification_KFold(X, y,
                                                                                   .3,  # Hyperparameters.olc_wa_w_inc,
                                                                                   Hyperparameters.olr_wa_base_model_size1,
                                                                                   Hyperparameters.olr_wa_increment_size(
                                                                                       n_features,
                                                                                       user_defined_val=10),
                                                                                   seed,
                                                                                   kpi=kpi,
                                                                                   multiplier=multiplier,
                                                                                   expr='expr11'
                                                                                   )
        end_time = time.perf_counter()
        olc_wa_sccm_execution_time += (end_time - start_time)
        olc_wa_sccm_acc_list_per_seed.append(olc_wa_sccm_acc)

        #
        # # 2. OLC_WA Experiment (One-Vs-One)
        # start_time = time.perf_counter()
        # olc_wa2_acc = OLCWA_MC_ONE_VS_ONE.olc_wa_mc_ovo_classification_KFold(X, y,
        #                                                                  .3,#Hyperparameters.olc_wa_w_inc,
        #                                                                  Hyperparameters.olr_wa_base_model_size1,
        #                                                                  Hyperparameters.olr_wa_increment_size(n_features,
        #                                                                                                user_defined_val=10),
        #                                                                  seed
        #                                                                  )
        # end_time = time.perf_counter()
        # olc_wa2_execution_time += (end_time - start_time)
        # olc_wa2_acc_list_per_seed.append(olc_wa2_acc)

        # 3. Passive Aggressive Experiment
        C = .001
        start_time = time.perf_counter()
        pa_acc = OnlinePassiveAggressive.online_passive_aggressive_multiclass_learn_KFold(X,y,C,seed)
        end_time = time.perf_counter()
        pa_execution_time += (end_time - start_time)
        pa_acc_list_per_seed.append(pa_acc)
        #
        # 4. Widrow-Hoff (ONE-VS-REST)
        learning_rate = 0.0001
        start_time = time.perf_counter()
        widrow_hoff_acc = WidrowHoff.widrow_hoff_ovr_KFold(X,y,learning_rate, seed)
        end_time = time.perf_counter()
        widrow_hoff_execution_time += (end_time - start_time)
        widrow_hoff_acc_list_per_seed.append(widrow_hoff_acc)
        #
        # 4. Widrow-Hoff (ONE-VS-ONE)
        learning_rate = 0.01
        start_time = time.perf_counter()
        widrow_hoff2_acc = WidrowHoff.widrow_hoff_ovo_KFold(X, y, learning_rate, seed)
        end_time = time.perf_counter()
        widrow_hoff2_execution_time += (end_time - start_time)
        widrow_hoff2_acc_list_per_seed.append(widrow_hoff2_acc)
        # #
        # 5. Perceptron (ONE-VS-REST)
        start_time = time.perf_counter()
        perceptron_acc = Perceptron.perceptron_ovr_KFold(X, y, seed)
        end_time = time.perf_counter()
        perceptron_execution_time += (end_time - start_time)
        perceptron_acc_list_per_seed.append(perceptron_acc)
        # # #
        # 5. Perceptron (ONE-VS-ONE)
        start_time = time.perf_counter()
        perceptron2_acc = Perceptron.perceptron_ovo_KFold(X, y, seed)
        end_time = time.perf_counter()
        perceptron2_execution_time += (end_time - start_time)
        perceptron2_acc_list_per_seed.append(perceptron2_acc)
        # #
        # 6. Online Logistic Regression (OVR)
        learning_rate = 0.001
        start_time = time.perf_counter()
        online_logistic_regression_acc = Online_Logistic_Regression.online_logistic_regression_ovr_KFold(X, y, learning_rate, seed)
        end_time = time.perf_counter()
        online_logistic_regression_execution_time += (end_time - start_time)
        online_logistic_regression_acc_list_per_seed.append(online_logistic_regression_acc)
        #
        # # # 6. Online Logistic Regression (OVO)
        # # learning_rate = 0.01
        # # start_time = time.perf_counter()
        # # online_logistic_regression2_acc = Online_Logistic_Regression.online_logistic_regression_ovo_KFold(X, y,
        # #                                                                                                  learning_rate,
        # #                                                                                                  seed)
        # # end_time = time.perf_counter()
        # # online_logistic_regression2_execution_time += (end_time - start_time)
        # # online_logistic_regression2_acc_list_per_seed.append(online_logistic_regression2_acc)
        # #
        # # 7. Online Naive Bayes
        # start_time = time.perf_counter()
        # online_naive_bayes_acc = OnlineNaiveBayes.online_naive_bayes_KFold(X, y, seed)
        # end_time = time.perf_counter()
        # online_naive_bayes_execution_time += (end_time - start_time)
        # online_naive_bayes_acc_list_per_seed.append(online_naive_bayes_acc)

        # 8. Hoeffding Tree
        start_time = time.perf_counter()
        hoeffding_tree_acc = HoeffdingTree.hoeffding_tree_KFold(X, y, seed)
        end_time = time.perf_counter()
        hoeffding_tree_execution_time += (end_time - start_time)
        hoeffding_tree_acc_list_per_seed.append(hoeffding_tree_acc)

        # 1. Results for Batch Experiment:
    batch_acc = np.array(batch_acc_list_per_seed).mean()
    print('Batch (Logistic Regression), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(batch_execution_time / total_runs), ', accuracy:', "{:.5f}".format(batch_acc))

    # 2. Results for OLC_WA Experiment (One-vs-Rest)
    olc_wa_acc = np.array(olc_wa_acc_list_per_seed).mean()
    print('OLC_WA (ovr), 5 folds, seeds averaging. time:', "{:.5f} s".format(olc_wa_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(olc_wa_acc))

    # 3. Results for OLC_WA (SCCM) Experiment (One-vs-Rest)
    olc_wa_sccm_acc = np.array(olc_wa_sccm_acc_list_per_seed).mean()
    print('OLC_WA (SCCM) (ovr), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(olc_wa_sccm_execution_time / total_runs), ', accuracy:',
          "{:.5f}".format(olc_wa_sccm_acc))

    # 2. Results for OLC_WA Experiment (One-vs-One)
    olc_wa2_acc = np.array(olc_wa2_acc_list_per_seed).mean()
    print('OLC_WA (ovo), 5 folds, seeds averaging. time:', "{:.5f} s".format(olc_wa2_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(olc_wa2_acc))

    # 2. Results for PA Experiment
    pa_acc = np.array(pa_acc_list_per_seed).mean()
    print('PA, 5 folds, seeds averaging. time:', "{:.5f} s".format(pa_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(pa_acc))

    # 4. Results for Widrow Hoff (OVR)
    widrow_hoff_acc = np.array(widrow_hoff_acc_list_per_seed).mean()
    print('Widrow-Hoff (OVR), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(widrow_hoff_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(widrow_hoff_acc))

    # 4. Results for Widrow Hoff (OVO)
    widrow_hoff2_acc = np.array(widrow_hoff2_acc_list_per_seed).mean()
    print('Widrow-Hoff (OVO), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(widrow_hoff2_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(widrow_hoff2_acc))

    # 4. Results for Perceptron (OVR)
    perceptron_acc = np.array(perceptron_acc_list_per_seed).mean()
    print('Perceptron (OVR), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(perceptron_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(perceptron_acc))

    # 4. Results for Perceptron (OVO)
    perceptron2_acc = np.array(perceptron2_acc_list_per_seed).mean()
    print('Perceptron (OVO), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(perceptron2_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(perceptron2_acc))

    # 5. Results for Online Logistic regression Experiment (OVR)
    online_logistic_regression_acc = np.array(online_logistic_regression_acc_list_per_seed).mean()
    print('Online Logistic Regression (OVR), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_logistic_regression_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(online_logistic_regression_acc))

    # 5. Results for Online Logistic regression Experiment (OVO)
    online_logistic_regression2_acc = np.array(online_logistic_regression2_acc_list_per_seed).mean()
    print('Online Logistic Regression (OVO), 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_logistic_regression2_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(online_logistic_regression2_acc))

    # 6. Results for Online Naive Bayes Experiment
    online_naive_bayes_acc = np.array(online_naive_bayes_acc_list_per_seed).mean()
    print('Online Naive Bayes, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(online_naive_bayes_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(online_naive_bayes_acc))

    # 7. Results for Hoeffding Tree Bayes Experiment
    hoeffding_tree_acc = np.array(hoeffding_tree_acc_list_per_seed).mean()
    print('Hoeffding Tree, 5 folds, seeds averaging. time:',
          "{:.5f} s".format(hoeffding_tree_execution_time / total_runs),
          ', accuracy:',
          "{:.5f}".format(hoeffding_tree_acc))


if __name__ == '__main__':
    experiment11Main()