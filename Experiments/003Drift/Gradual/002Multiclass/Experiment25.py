import numpy as np

from Datasets.Drift.Gradual.Multiclass import DS25
import time
from Models.OLCWA import OLCWA, OLCWA_MC_ONE_VS_REST, OLCWA_MC_ONE_VS_ONE, OLCWA_MC_ONE_VS_REST_LATEST
from Models.Perceptron import Perceptron
from Models.WidrowHoff import WidrowHoff
from Models.Online_Logistic_Regression import Online_Logistic_Regression
from Models.OnlineNaiveBayes import OnlineNaiveBayes
from Models.HoeffdingTree import HoeffdingTree
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Hyperparameters import Hyperparameters
from Utils import Printer, Predictions, Measures, PredictionsMultiClass
from Utils import Constants
import matplotlib.pyplot as plt
from Utils import Constants
import seaborn as sns
from Utils import Plotter
from matplotlib.lines import Line2D


def experiment25Main():
    X, y = DS25.get_DS25()
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    print("n_samples",X.shape[0], "n_features ", X.shape[1], "n_classes", n_classes)

    train_percent = int(92 * n_samples / 100)
    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]


    #
    # 1. OLC_WA Experiment
    print("OLC-WA LSTM_SCCM (OVR)  : Dynamic Hyperparameters, Start with W_inc = 0.5, W_base = 0.5")
    start_time = time.perf_counter()
    kpi = 'ACC'
    multiplier_acc = 1.5
    olc_wa_lstm_sccm_ovr_coeff_classifiers, olc_wa_lstm_sccm_ovr_epoch_list, olc_wa_lstm_sccm_ovr_cost_list, \
        olc_wa_lstm_sccm_ovr_acc_list = \
        OLCWA_MC_ONE_VS_REST_LATEST.olc_wa_lstm_sccm_ovr_classification(X_train, y_train,
                                                                 .3, #Hyperparameters.olc_wa_w_inc,
                                                                 Hyperparameters.olr_wa_base_model_size1,
                                                                 150,
                                                                 # Hyperparameters.olr_wa_increment_size2(n_features,
                                                                 #                                 user_defined_val=10)
                                                                 kpi=kpi, multiplier=multiplier_acc, expr='expr21')
    end_time = time.perf_counter()
    olc_wa_lstm_sccm_ovr_execution_time = (end_time - start_time)
    Printer.print_list_tabulate(olc_wa_lstm_sccm_ovr_acc_list)
    olc_wa_lstm_sccm_ovr_acc = PredictionsMultiClass.compute_acc_olr_wa_ovr(X_test, y_test,
                                                                            olc_wa_lstm_sccm_ovr_coeff_classifiers)
    # print('olc_wa_lstm_sccm_ovr_epoch_list', olc_wa_lstm_sccm_ovr_epoch_list)
    print('OLC-WA LSTM_SCCM (OVR). time:', "{:.5f} s".format(olc_wa_lstm_sccm_ovr_execution_time), ', accuracy:',
          "{:.5f}".format(olc_wa_lstm_sccm_ovr_acc))
    print("End of OLC-WA LSTM_SCCM (OVR) ")
    print('********************************************************************************************')



    print("Passive Aggressive (PA)")
    start_time = time.perf_counter()
    C = 1
    pa_w, pa_b, pa_epoch_list, pa_cost_list, pa_acc_list = OnlinePassiveAggressive.online_passive_aggressive_multiclass_sklearn2(
        X_train, y_train, C, record_cost_on_every_epoch=150)
    pa_predicted_y_test = Predictions.predict_multiclass2(X_test, pa_w,
                                                          pa_b)  # make sure the order of w is same as coeff
    pa_acc = Measures.accuracy(y_test, pa_predicted_y_test)
    end_time = time.perf_counter()
    pa_execution_time = (end_time - start_time)
    # print('pa_epoch_list ', pa_epoch_list)
    print('pa_acc_list', pa_acc_list)
    print('PA . time:', "{:.5f} s".format(pa_execution_time), ', accuracy:',
          "{:.5f}".format(pa_acc))
    print("End of PA")

    print('********************************************************************************************')

    print("Widrow-Hoff (LMS)")
    start_time = time.perf_counter()
    learning_rate = 0.1
    lms_classifiers, lms_epoch_lists, lms_cost_lists, lms_acc_list = WidrowHoff.widrow_hoff_mc_ovr(X_train, y_train,
                                                                                                   learning_rate,
                                                                                                   recod_cost_at_each_epochs=150)
    # comupute acc for test data:
    lms_acc = Measures.widrow_hoff_acc_mc_ovr(lms_classifiers, X_test, y_test, n_classes)
    end_time = time.perf_counter()
    lms_execution_time = (end_time - start_time)
    print('lms_epoch_lists', lms_epoch_lists)
    print('lms_acc_list', lms_acc_list)
    print('LMS . time:', "{:.5f} s".format(lms_execution_time), ', accuracy:',
          "{:.5f}".format(lms_acc))
    print("End of LMS")

    print('********************************************************************************************')
    print("Perceptron")
    start_time = time.perf_counter()
    perceptron_classifiers, perceptron_epoch_lists, perceptron_cost_lists, perceptron_acc_list = \
        Perceptron.perceptron_mc_ovr(X_train, y_train, record_cost_at_each_epochs=150)
    # compute acc for test data:
    perceptron_acc = Measures.perceptron_acc_mc_ovr(perceptron_classifiers, X_test, y_test)
    # perceptron_acc = Measures.predict_perceptron_mc(perceptron_classifiers, X_test, y_test,n_classes)
    end_time = time.perf_counter()
    perceptron_execution_time = (end_time - start_time)
    print('perceptron_epoch_lists', perceptron_epoch_lists)
    print('perceptron_acc_list', perceptron_acc_list)
    print('Perceptron . time:', "{:.5f} s".format(perceptron_execution_time), ', accuracy:',
          "{:.5f}".format(perceptron_acc))
    print("End of Perceptron")
    print('********************************************************************************************')
    #
    # 6. Online Logistic Regression
    learning_rate = 0.1
    start_time = time.perf_counter()
    olr_classifiers, olr_epoch_lists, olr_cost_lists,  olr_acc_lists = Online_Logistic_Regression.online_logistic_regression_ovr_final(
        X_train, y_train, learning_rate, record_cost_on_every_epoch=150)
    end_time = time.perf_counter()
    predicted_y_test = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        # Get the predicted probabilities for each classifier
        probabilities = np.array(
            [Predictions.predict_prob_2((X_test[i].reshape(1, -1)), clf[0], clf[1]) for clf in olr_classifiers])
        predicted_y_test[i] = np.argmax(probabilities)
    online_logistic_regression_acc = Measures.accuracy(y_test, predicted_y_test)

    online_logistic_regression_execution_time = (end_time - start_time)
    Printer.print_list_tabulate(olr_acc_lists)
    print('Online Logistic Regression. time:', "{:.5f} s".format(online_logistic_regression_execution_time),
          ', accuracy:', "{:.5f}".format(online_logistic_regression_acc))

    print('********************************************************************************************')

    print("Naive Bayes")
    start_time = time.perf_counter()
    nb_classifier, nb_epoch_lists, nb_cost_lists, nb_acc_list = \
        OnlineNaiveBayes.online_naive_bayes(X_train, y_train, record_cost_on_every_epoch=150)
    # compute acc for test data:
    predicted_y_test = Predictions.predict_naive_bayes1(X_test, nb_classifier)
    nb_acc = Measures.accuracy(y_test, predicted_y_test)
    end_time = time.perf_counter()
    nb_execution_time = (end_time - start_time)
    print('nb_epoch_lists', nb_epoch_lists)
    print('nb_acc_list', nb_acc_list)
    print('Naive Bayes . time:', "{:.5f} s".format(nb_execution_time), ', accuracy:',
          "{:.5f}".format(nb_acc))
    print("End of naive Bayes")
    print('********************************************************************************************')

    print("Hoeffding Tree")
    start_time = time.perf_counter()
    ht_classifier, ht_epoch_lists, ht_cost_lists, ht_acc_list = \
         HoeffdingTree.hoeffding_tree(X_train, y_train, record_cost_on_every_epoch=150)
    # compute acc for test data:
    y_predicted = Predictions.predict_hoeffding_tree(X_test, ht_classifier)
    ht_acc = Measures.accuracy(y_test, y_predicted)
    end_time = time.perf_counter()
    ht_execution_time = (end_time - start_time)
    print('ht_epoch_lists', ht_epoch_lists)
    print('ht_acc_list', ht_acc_list)
    print('Hoeffding Tree. time:', "{:.5f} s".format(ht_execution_time), ', accuracy:',
          "{:.5f}".format(ht_acc))
    print("End of Hoeffding Tree")
    print('********************************************************************************************')

    print('Plotting')
    plotting_enabled = True
    if (plotting_enabled):

        x_axis_olc_wa_sccm = olc_wa_lstm_sccm_ovr_epoch_list
        y_axis_olc_wa_sccm = olc_wa_lstm_sccm_ovr_acc_list
        x_axis_pa = pa_epoch_list
        y_axis_pa = pa_acc_list
        x_axis_widrow_hoff = lms_epoch_lists
        y_axis_widrow_hoff = lms_acc_list
        x_axis_perceptron = perceptron_epoch_lists
        y_axis_perceptron = perceptron_acc_list
        x_axis_online_logistic_regression = olr_epoch_lists
        y_axis_online_logistic_regression = olr_acc_lists
        x_axis_naive_bayes = nb_epoch_lists
        y_axis_naive_bayes = nb_acc_list
        x_axis_hoeffding_tree = ht_epoch_lists
        y_axis_hoeffding_tree = ht_acc_list

        # x_axis_olc_wa_sccm = olc_wa_lstm_sccm_ovr_epoch_list
        # y_axis_olc_wa_sccm = olc_wa_lstm_sccm_ovr_acc_list
        # x_axis_pa = [] # pa_epoch_list
        # y_axis_pa = [] # pa_acc_list
        # x_axis_widrow_hoff = [] # lms_epoch_lists
        # y_axis_widrow_hoff = [] # lms_acc_list
        # x_axis_perceptron = [] # perceptron_epoch_lists
        # y_axis_perceptron = [] # perceptron_acc_list
        # x_axis_online_logistic_regression = [] #olr_epoch_lists
        # y_axis_online_logistic_regression = [] #olr_acc_lists
        # x_axis_naive_bayes = [] # nb_epoch_lists
        # y_axis_naive_bayes = [] # nb_acc_list
        # x_axis_hoeffding_tree = [] # ht_epoch_lists
        # y_axis_hoeffding_tree = [] # ht_acc_list

        kpi = 'ACC'
        label_olc_wa = 'OLC-WA'
        label_pa = 'PA'
        label_lms = 'LMS'
        label_pla = 'PLA'
        label_olr = 'OLR'
        label_nb = 'N.B.'
        label_ht = 'VFDT'

        plot_results( x_axis_olc_wa_sccm, y_axis_olc_wa_sccm, x_axis_pa, y_axis_pa,
                     x_axis_widrow_hoff, y_axis_widrow_hoff, x_axis_perceptron, y_axis_perceptron,
                     x_axis_online_logistic_regression, y_axis_online_logistic_regression,
                     x_axis_naive_bayes, y_axis_naive_bayes, x_axis_hoeffding_tree, y_axis_hoeffding_tree,
                     kpi, label_olc_wa, label_pa, label_lms, label_pla, label_olr, label_nb, label_ht,
                     drift_location=None, log_enabled=False,
                     legend_loc='lower left',
                      drift_type='gradual',
                      gradual_drift_locations = [1200 , 1500 , 1800,  2400 , 2700 , 3900],
                      gradual_drift_concepts=['c2', 'c1', 'c2', 'c1', 'c2', 'c1']
                      )

def plot_results(x_axis_olc_wa_sccm, y_axis_olc_wa_sccm, x_axis_pa, y_axis_pa,
                 x_axis_widrow_hoff, y_axis_widrow_hoff, x_axis_perceptron, y_axis_perceptron,
                 x_axis_online_logistic_regression, y_axis_online_logistic_regression,
                 x_axis_naive_bayes, y_axis_naive_bayes, x_axis_hoeffding_tree, y_axis_hoeffding_tree,
                 kpi, label_olc_wa, label_pa, label_lms, label_pla, label_olr, label_nb, label_ht,
                 drift_location, log_enabled,
                 legend_loc, drift_type, gradual_drift_locations, gradual_drift_concepts):

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    print('x_axis_olc_wa_sccm: ',x_axis_olc_wa_sccm)
    print('y_axis_olc_wa_sccm: ',y_axis_olc_wa_sccm)

    line1, = plt.plot(x_axis_olc_wa_sccm, y_axis_olc_wa_sccm, linestyle='-', marker='o', markersize=3, linewidth=2,
                      label=label_olc_wa, zorder=10)

    line2, = plt.plot(x_axis_pa, y_axis_pa, linestyle='-', marker='.', markersize=1, linewidth=0.9, label=label_pa,
                      zorder=9)
    line3, = plt.plot(x_axis_widrow_hoff, y_axis_widrow_hoff, linestyle='-', marker='.', markersize=1, linewidth=0.9,
                      label=label_lms)
    line4, = plt.plot(x_axis_perceptron, y_axis_perceptron, linestyle='-', marker='.', markersize=1, linewidth=0.9,
                      label=label_pla)
    line5, = plt.plot(x_axis_online_logistic_regression, y_axis_online_logistic_regression, linestyle='-', marker='.',
                      markersize=1, linewidth=0.9, label=label_olr)
    line6, = plt.plot(x_axis_naive_bayes, y_axis_naive_bayes, linestyle='-', marker='.', markersize=1, linewidth=0.9,
                      label=label_nb)
    line7, = plt.plot(x_axis_hoeffding_tree, y_axis_hoeffding_tree, linestyle='-', marker='.', markersize=1,
                      linewidth=0.9, label=label_ht)


    if drift_type == 'abrupt':
        # Shade the region from x=500 onwards
        plt.axvspan(drift_location, max(x_axis_olc_wa_sccm), color=Constants.color_yello, alpha=0.3,
                    label='Concept Drift')

    # if drift_type == 'incremental':
    #     # Draw a vertical yellow line at each drift_location   max(x_axis)
    #     for loc in range(drift_location, int(max(x_axis_olc_wa_sccm)) , drift_location):
    #         plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=.6, label=None)
    #
    #     concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
    #                                 label='Concept Drift')

    # if drift_type == 'incremental':
    #     # gradual_drift_location = [250, 350, 450, 650, 750, 1000]
    #     # concepts = ['c1', 'c2', 'c1', 'c2', 'c1', 'c2']
    #     # concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
    #     for loc in zip(incremental_drift_locations):
    #         # color = concept_colors.get(concept, Constants.color_blue)  # Default color if concept not found
    #         plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=0.6)
    #
    #     concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
    #                                                                 label='Concept Drift')

    if drift_type == 'gradual':
        concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc, concept in zip(gradual_drift_locations[:-1], gradual_drift_concepts[:-1]):
            color = concept_colors.get(concept, Constants.color_blue)  # Default color if concept not found
            plt.axvline(x=loc, color=color, linestyle='-', linewidth=0.6)

        # Create custom legend entries
        gradual_concept_drift_line = [
            Line2D([0], [0], color=color, linestyle='-', linewidth=0.6, label=f'Concept {concept}')
            for concept, color in concept_colors.items()]



    # Adding labels and title
    plt.xlabel('$N$', fontsize=7)

    if kpi == 'ACC': plt.ylabel('ACC', fontsize=7)
    if kpi == 'COST': plt.ylabel('COST', fontsize=7)

    plt.title('Performance Comparison', fontsize=7)

    # Adjust font size of numbers on x-axis and y-axis
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)

    # Customize grid
    plt.grid(axis='x', linestyle='--', alpha=0.7)  # Only vertical lines with dashed style

    # Remove top and right spines
    sns.despine()

    # Adding grid
    plt.grid(True)

    if drift_type == 'gradual':
        legend = plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
                   fontsize='x-small',
                   loc=legend_loc, fancybox=True, shadow=True,
                   borderpad=1, labelspacing=0,
                   facecolor='lightblue', edgecolor=Constants.color_black)
        legend.set_zorder(15)
    # Show plot
    plt.show()


if __name__ == "__main__":
    experiment25Main()
