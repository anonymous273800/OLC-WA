import warnings
import numpy as np
import matplotlib.pyplot as plt
from Utils import Constants
import seaborn as sns
from matplotlib.lines import Line2D
import Hyperparameters.Hyperparameters
import time
from Models.OLCWA import OLCWA, OLCWAFinal
from Models.OnlinePassiveAggressive import OnlinePassiveAggressive
from Models.WidrowHoff import WidrowHoff
from Hyperparameters import Hyperparameters
from Models.Perceptron import Perceptron
from Models.Online_Logistic_Regression import Online_Logistic_Regression
from Models.OnlineNaiveBayes import OnlineNaiveBayes
from Models.HoeffdingTree import HoeffdingTree
from Datasets.Drift.Incremental.Binary import DS20
from Utils import Predictions, Measures, Printer

from Utils import Plotter
warnings.filterwarnings("ignore")

def experiment19Main():
    # Abrupt Drift dataset
    X, y = DS20.get_DS20()
    n_samples, n_features = X.shape
    print(n_samples, n_features)
    train_percent = int(95 * n_samples / 100)

    X_train = X[:train_percent]
    y_train = y[:train_percent]
    X_test = X[train_percent:]
    y_test = y[train_percent:]

    # # 1. OLC_WA Experiment
    # print("OLC-WA : Fixed Hyperparameters W_inc = 0.5, W_base = 0.5")
    # start_time = time.perf_counter()
    # olr_wa_coeff, olr_wa_epoch_list, olr_wa_cost_list, olr_wa_acc_list = \
    #     OLCWA.olc_wa_classification(X_train, y_train,
    #                               Hyperparameters.olc_wa_w_inc,
    #                               Hyperparameters.olr_wa_base_model_size1,
    #                               Hyperparameters.olr_wa_increment_size2(n_features,
    #                                                                   user_defined_val=10)
    #                               )
    #
    # end_time = time.perf_counter()
    # predicted_y_test = Predictions.predict_(X_test, olr_wa_coeff)
    # olc_wa_acc = Measures.accuracy(y_test, predicted_y_test)
    # olc_wa_execution_time = (end_time - start_time)
    # Printer.print_list(olr_wa_acc_list, kpi='accuracy')
    # # Printer.print_list(olr_wa_cost_list, kpi='cross-entopy')
    # Printer.print_list_tabulate(olr_wa_acc_list)
    # print('OLC_WA. time:', "{:.5f} s".format(olc_wa_execution_time), ', accuracy:',
    #       "{:.5f}".format(olc_wa_acc))

    print("*------------------------------------------------------------------------------------------------------")
    # 2. OLC_WA (with LSTM-SCCM) Experiment
    print("OLC-WA (SCCM): Start with Fixed Hyperparameters W_inc = 0.5, W_base = 0.5")
    kpi = 'ACC'
    multiplier_acc = 2.5
    start_time = time.perf_counter()
    olr_wa_lstm_sccm_coeff, olr_wa_lstm_sccm_epoch_list, olr_wa_lstm_sccm_cost_list, olr_wa_lstm_sccm_acc_list = \
                                                OLCWAFinal.olc_wa_lstm_sccm_classification(X_train, y_train,
                                                               .5,#Hyperparameters.olc_wa_w_inc,
                                                               #Hyperparameters.olr_wa_base_model_size1,
                                                               200, #Hyperparameters.olr_wa_increment_size(n_features,
                                                                 #                               user_defined_val=10),
                                                               kpi=kpi,
                                                               multiplier=multiplier_acc,
                                                               expr='expr20'
                                                               )
    predicted_y_test = Predictions.predict_(X_test, olr_wa_lstm_sccm_coeff)
    olc_wa_lstm_sccm_acc = Measures.accuracy(y_test, predicted_y_test)
    end_time = time.perf_counter()
    olc_wa_lstm_sccm_execution_time=(end_time - start_time)
    Printer.print_list(olr_wa_lstm_sccm_acc_list, kpi='accuracy')
    # Printer.print_list(olr_wa_lstm_sccm_cost_list, kpi='cross-entopy')
    Printer.print_list_tabulate(olr_wa_lstm_sccm_acc_list)
    print('OLC_WA LSTM_SCCM. time:', "{:.5f} s".format(olc_wa_lstm_sccm_execution_time),', accuracy:', "{:.5f}".format(olc_wa_lstm_sccm_acc))

    print("------------------------------------------------------------------------------------------------------")
    # 3. Passive-Aggressive Experiment
    C = .1
    start_time = time.perf_counter()
    pa_w, pa_b, pa_epoch_list, pa_cost_list, pa_acc_list = OnlinePassiveAggressive.online_passive_aggressive_binary(X_train, y_train, C, record_cost_on_every_epoch=200)
    pa_predicted_y_test = Predictions.predict(X_test, pa_w, pa_b)  # make sure the order of w is same as coeff
    pa_acc = Measures.accuracy(y_test, pa_predicted_y_test)
    end_time = time.perf_counter()
    pa_execution_time = (end_time - start_time)
    Printer.print_list_tabulate(pa_acc_list)
    print('PA. time:', "{:.5f} s".format(pa_execution_time),
          ', accuracy:', "{:.5f}".format(pa_acc))

    print("------------------------------------------------------------------------------------------------------")
    # 4. WidrowHoff
    learning_rate = 0.01
    start_time = time.perf_counter()
    w, b, widrow_hoff_epoch_list,widrow_hoff_cost_list, widrow_hoff_acc_list = WidrowHoff.widrow_hoff_classification(X_train, y_train, learning_rate, record_cost_on_every_epoch=200)
    widrow_hoff_y_predicted = Predictions.predict_widrow_hoff(X_test, w, b)
    widrow_hoff_acc = Measures.accuracy(y_test, widrow_hoff_y_predicted)
    end_time = time.perf_counter()
    widrow_hoff_execution_time = (end_time - start_time)
    Printer.print_list_tabulate(widrow_hoff_acc_list)
    print('Widrow Hoff. time:', "{:.5f} s".format(widrow_hoff_execution_time),
          ', accuracy:', "{:.5f}".format(widrow_hoff_acc))

    print("------------------------------------------------------------------------------------------------------")
    # 5. Perceptron
    start_time = time.perf_counter()
    perceptron_w, perceptron_b, perceptron_epoch_list, perceptron_cost_list, perceptron_acc_list  = \
                                                        Perceptron.perceptron(X_train, y_train, record_cost_on_every_epoch=200)
    end_time = time.perf_counter()
    perceptron_execution_time =(end_time - start_time)
    perceptron_y_predicted = Predictions.predict_widrow_hoff(X_test, perceptron_w, perceptron_b)  # TODO: check this
    perceptron_acc = Measures.accuracy(y_test, perceptron_y_predicted)
    Printer.print_list_tabulate(perceptron_acc_list)
    print('Perceptron. time:', "{:.5f} s".format(perceptron_execution_time),
          ', accuracy:', "{:.5f}".format(perceptron_acc))

    # print("------------------------------------------------------------------------------------------------------")
    # 6. Online Logistic Regression
    learning_rate = 0.1
    start_time = time.perf_counter()
    online_logistic_regression_w, online_logistic_regression_b, online_logistic_regression_epoch_list, \
        online_logistic_regression_cost_list, online_logistic_regression_acc_list = Online_Logistic_Regression.online_logistic_regression(X_train, y_train, learning_rate, record_cost_on_every_epoch=200)
    end_time = time.perf_counter()
    online_logistic_regression_y_predicted = Predictions.predict(X_test, online_logistic_regression_w, online_logistic_regression_b)
    online_logistic_regression_acc = Measures.accuracy(y_test, online_logistic_regression_y_predicted)
    online_logistic_regression_execution_time = (end_time - start_time)
    Printer.print_list_tabulate(online_logistic_regression_acc_list)
    print('Online Logistic Regression. time:', "{:.5f} s".format(online_logistic_regression_execution_time),
          ', accuracy:', "{:.5f}".format(online_logistic_regression_acc))

    #
    print("------------------------------------------------------------------------------------------------------")
    # 7. Online Naive Bayes
    start_time = time.perf_counter()
    naive_bayes_model, naive_bayes_epoch_list, naive_bayes_cost_list, naive_bayes_acc_list = \
        OnlineNaiveBayes.online_naive_bayes(X_train, y_train, record_cost_on_every_epoch=200)
    end_time = time.perf_counter()
    online_naive_bayes_execution_time = (end_time - start_time)
    online_naive_bayes_predicted_y_test = Predictions.predict_naive_bayes1(X_test, naive_bayes_model)
    online_naive_bayes_execution_acc = Measures.accuracy(y_test, online_naive_bayes_predicted_y_test)
    Printer.print_list_tabulate(naive_bayes_acc_list)
    print('Online Naive Bayes. time:', "{:.5f} s".format(online_naive_bayes_execution_time),
          ', accuracy:', "{:.5f}".format(online_naive_bayes_execution_acc))

    print("------------------------------------------------------------------------------------------------------")
    # 8. Hoeffding Tree
    start_time = time.perf_counter()
    hoeffding_tree_model, hoeffding_tree_epoch_list, hoeffding_tree_cost_list, hoeffding_tree_acc_list = HoeffdingTree.hoeffding_tree(X_train, y_train, record_cost_on_every_epoch=200)
    end_time = time.perf_counter()
    hoeffding_tree_execution_time = (end_time - start_time)

    hoeffding_tree_model_y_predicted = []
    for i in range(len(X_test)):
        x_test_instance = {f'feature_{j}': X_test[i, j] for j in range(n_features)}
        y_test_pred = hoeffding_tree_model.predict_one(x_test_instance)
        hoeffding_tree_model_y_predicted.append(y_test_pred)
    hoeffding_tree_acc = Measures.accuracy(y_test, hoeffding_tree_model_y_predicted)
    Printer.print_list_tabulate(hoeffding_tree_acc_list)
    print('Hoeffding Tree. time:', "{:.5f} s".format(hoeffding_tree_execution_time),
          ', accuracy:', "{:.5f}".format(hoeffding_tree_acc))

    print('Plotting')
    plotting_enabled = True
    if (plotting_enabled):
        # mini_batch_size = Hyperparameters.olr_wa_increment_size2(n_features, user_defined_val=10)
        # start = int(X_train.shape[0] * 10/100) # no of points for base model as the first acc for the base model, should align the no of points with the acc
        # x_axis = [i for i in range(start, n_samples + mini_batch_size, mini_batch_size)]
        # x_axis = [i for i in range(mini_batch_size, n_samples + mini_batch_size, mini_batch_size)]

        # # x_axis_olr_wa = olr_wa_epoch_list
        # # y_axis_olr_wa = olr_wa_acc_list

        x_axis_olc_wa_sccm = olr_wa_lstm_sccm_epoch_list
        y_axis_olc_wa_sccm = olr_wa_lstm_sccm_acc_list
        x_axis_pa = pa_epoch_list
        y_axis_pa = pa_acc_list
        x_axis_widrow_hoff = widrow_hoff_epoch_list
        y_axis_widrow_hoff = widrow_hoff_acc_list
        x_axis_perceptron = perceptron_epoch_list
        y_axis_perceptron = perceptron_acc_list
        x_axis_online_logistic_regression = online_logistic_regression_epoch_list
        y_axis_online_logistic_regression = online_logistic_regression_acc_list
        x_axis_naive_bayes = naive_bayes_epoch_list
        y_axis_naive_bayes = naive_bayes_acc_list
        x_axis_hoeffding_tree = hoeffding_tree_epoch_list
        y_axis_hoeffding_tree = hoeffding_tree_acc_list

        # x_axis_olc_wa_sccm = olr_wa_lstm_sccm_epoch_list
        # y_axis_olc_wa_sccm = olr_wa_lstm_sccm_acc_list
        # x_axis_pa = []
        # y_axis_pa = []
        # x_axis_widrow_hoff = []
        # y_axis_widrow_hoff = []
        # x_axis_perceptron = []
        # y_axis_perceptron = []
        # x_axis_online_logistic_regression = []
        # y_axis_online_logistic_regression = []
        # x_axis_naive_bayes = []#naive_bayes_epoch_list
        # y_axis_naive_bayes = [] #naive_bayes_acc_list
        # x_axis_hoeffding_tree = [] #hoeffding_tree_epoch_list
        # y_axis_hoeffding_tree = [] #hoeffding_tree_acc_list

        kpi = 'ACC'
        label_olc_wa = 'OLC-WA'
        label_pa = 'PA'
        label_lms = 'LMS'
        label_pla = 'PLA'
        label_olr = 'OLR'
        label_nb = 'N.B.'
        label_ht = 'VFDT'
        plot_results(x_axis_olc_wa_sccm, y_axis_olc_wa_sccm, x_axis_pa, y_axis_pa,
                     x_axis_widrow_hoff, y_axis_widrow_hoff, x_axis_perceptron, y_axis_perceptron,
                     x_axis_online_logistic_regression, y_axis_online_logistic_regression,
                     x_axis_naive_bayes, y_axis_naive_bayes, x_axis_hoeffding_tree, y_axis_hoeffding_tree,
                     kpi, label_olc_wa, label_pa, label_lms, label_pla, label_olr, label_nb, label_ht,
                     drift_location=100, log_enabled=False,
                     legend_loc='lower left', drift_type='incremental',
                     gradual_drift_locations=[2250, 3750, 5250, 6750, 8250, 9750, 11250, 12750, 14250])


def plot_results(x_axis_olc_wa_sccm, y_axis_olc_wa_sccm, x_axis_pa, y_axis_pa,
                 x_axis_widrow_hoff, y_axis_widrow_hoff, x_axis_perceptron, y_axis_perceptron,
                 x_axis_online_logistic_regression, y_axis_online_logistic_regression,
                 x_axis_naive_bayes, y_axis_naive_bayes, x_axis_hoeffding_tree, y_axis_hoeffding_tree,
                 kpi, label_olc_wa, label_pa, label_lms, label_pla, label_olr, label_nb, label_ht,
                 drift_location, log_enabled,
                 legend_loc, drift_type, gradual_drift_locations):

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
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

    if drift_type == 'incremental':
        # gradual_drift_location = [250, 350, 450, 650, 750, 1000]
        # concepts = ['c1', 'c2', 'c1', 'c2', 'c1', 'c2']
        # concept_colors = {'c1': Constants.color_yello, 'c2': Constants.color_green}
        for loc in zip(gradual_drift_locations):
            # color = concept_colors.get(concept, Constants.color_blue)  # Default color if concept not found
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=0.6)

        concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
                                                                    label='Concept Drift')



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

    # Adding legend
    if drift_type == 'incremental':
        legend = plt.legend(handles=[line1, line2, line3, line4, line5, line6, line7, concept_drift_line],
                   fontsize='x-small',
                   loc=legend_loc, fancybox=True, shadow=True,
                   borderpad=1, labelspacing=0,
                   facecolor='lightblue', edgecolor=Constants.color_black)
        legend.set_zorder(15)
    # Show plot
    plt.show()



if __name__ == '__main__':
    experiment19Main()