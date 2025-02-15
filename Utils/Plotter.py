import numpy as np
import matplotlib.pyplot as plt
from Utils import Constants
import seaborn as sns
from matplotlib.lines import Line2D

def plot(X, y, coeff):
    # Extract the feature vectors from X
    X1 = X[:, 0]
    X2 = X[:, 1]

    # Create the scatter plot for the training data
    plt.figure(figsize=(8, 6))
    plt.scatter(X1, X2, c=y, cmap='bwr', marker='o', edgecolors='k', label='Training Data')

    # Plot decision boundary
    x_vals = np.linspace(X1.min(), X1.max(), 100)
    # Decision boundary equation: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b)/w2
    w1, w2 = coeff[:2]  # coefficients for features
    b = coeff[-1]  # bias term
    y_vals = -(w1 * x_vals + b) / w2
    plt.plot(x_vals, y_vals, 'g--', label='Decision Boundary')

    # Labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data and Decision Boundary')
    plt.legend()
    plt.show()



# def plot_results(x_axis_olr_wa, y_axis_olr_wa, x_axis_olr_wa_lstm_sccm, y_axis_olr_wa_lstm_sccm,
#                  x_axis_pa, y_axis_pa,
#                  x_axis_widrow_hoff, y_axis_widrow_hoff, x_axis_perceptron, y_axis_perceptron,
#                  x_axis_online_logistic_regression, y_axis_online_logistic_regression,
#                  x_axis_naive_bayes, y_axis_naive_bayes, x_axis_hoeffding_tree, y_axis_hoeffding_tree,
#                  kpi, label1, label2, label3, label4, label5, label6, label7, label8,
#                  drift_location, log_enabled,
#                  legend_loc, drift_type, gradual_drift_locations, gradual_drift_concepts):
def plot_results(x_axis_olr_wa, y_axis_olr_wa, x_axis_olr_wa_lstm_sccm, y_axis_olr_wa_lstm_sccm,
                 x_axis_olr_wa_sccm, y_axis_olr_wa_sccm,
                 x_axis_pa, y_axis_pa,
                 kpi, label1, label2, label3, label4, label5, label6, label7, label8,
                 drift_location, log_enabled,
                 legend_loc, drift_type, gradual_drift_locations, gradual_drift_concepts):

    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    # line1, = plt.plot(x_axis_olr_wa, y_axis_olr_wa, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label1)
    # line2, = plt.plot(x_axis_olr_wa_lstm_sccm, y_axis_olr_wa_lstm_sccm, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label2)
    line3, = plt.plot(x_axis_olr_wa_sccm, y_axis_olr_wa_sccm, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label3)
    line4, = plt.plot(x_axis_pa, y_axis_pa, linestyle='-', marker='.', markersize=1, linewidth=0.8,label=label4)

    # line3, = plt.plot(x_axis_pa, y_axis_pa, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label3)
    # line4, = plt.plot(x_axis_widrow_hoff, y_axis_widrow_hoff, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label4)
    # line5, = plt.plot(x_axis_perceptron, y_axis_perceptron, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label5)
    # line6, = plt.plot(x_axis_online_logistic_regression, y_axis_online_logistic_regression, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label6)
    # line7, = plt.plot(x_axis_naive_bayes, y_axis_naive_bayes, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label7)
    # line8, = plt.plot(x_axis_hoeffding_tree, y_axis_hoeffding_tree, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label8)

    if drift_type == 'abrupt':
        # Shade the region from x=500 onwards
        plt.axvspan(drift_location, max(x_axis_olr_wa), color=Constants.color_yello, alpha=0.3, label='Concept Drift')

    if drift_type == 'incremental':
        # Draw a vertical yellow line at each drift_location
        for loc in range(drift_location, max(x_axis_olr_wa), drift_location):
            plt.axvline(x=loc, color=Constants.color_yello, linestyle='-', linewidth=.6, label=None)

        concept_drift_line = Line2D([0], [0], color=Constants.color_yello, linestyle='-', linewidth=.6,
                                    label='Concept Drift')

    if drift_type == 'gradual':
        # gradual_drift_location = [250, 350, 450, 650, 750, 1000]
        # concepts = ['c1', 'c2', 'c1', 'c2', 'c1', 'c2']
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

    if kpi == 'R2': plt.ylabel('R$^2$', fontsize=7)
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

    # # Adding legend
    # if drift_type == 'incremental':
    #     plt.legend(handles=[line1, line2, concept_drift_line], fontsize='small', loc=legend_loc, fancybox=True,
    #                shadow=True,
    #                borderpad=1, labelspacing=.5,
    #                facecolor=Constants.color_light_blue, edgecolor=Constants.color_black)
    #
    #     # Adding legend
    # if drift_type == 'gradual':
    #     plt.legend(handles=[line1, line2, gradual_concept_drift_line[0], gradual_concept_drift_line[1]],
    #                fontsize='small',
    #                loc=legend_loc, fancybox=True, shadow=True,
    #                borderpad=1, labelspacing=.5,
    #                facecolor='lightblue', edgecolor=Constants.color_black)

    if drift_type == 'abrupt':
        plt.legend(fontsize='small', loc=legend_loc, fancybox=True, shadow=True, borderpad=1, labelspacing=.5,
                   facecolor='lightblue', edgecolor=Constants.color_black)

    # Show plot
    plt.show()




def plot_results_experimental(x1, y1, x2, y2, x3, y3, x4, y4,x5,y5,x6, y6, x7,y7, x8, y8,
                             kpi, label1, label2, label3, label4,label5, label6, label7, label8,
                 drift_location, log_enabled,
                 legend_loc, drift_type, gradual_drift_locations, gradual_drift_concepts):


    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    line1, = plt.plot(x1, y1, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label1)
    line2, = plt.plot(x2, y2, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label2)
    line3, = plt.plot(x3, y3, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label3)
    line4, = plt.plot(x4, y4, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label4)
    line5, = plt.plot(x5, y5, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label5)
    line6, = plt.plot(x6, y6, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label6)
    line7, = plt.plot(x7, y7, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label7)
    line8, = plt.plot(x8, y8, linestyle='-', marker='.', markersize=1, linewidth=0.8, label=label8)


    if drift_type == 'abrupt':
        # Shade the region from x=500 onwards
        plt.axvspan(drift_location, max(x1), color=Constants.color_yello, alpha=0.3, label='Concept Drift')


    # Adding labels and title
    plt.xlabel('$N$', fontsize=7)

    if kpi == 'R2': plt.ylabel('R$^2$', fontsize=7)
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



    if drift_type == 'abrupt':
        plt.legend(fontsize='small', loc=legend_loc, fancybox=True, shadow=True, borderpad=1, labelspacing=.5,
                   facecolor='lightblue', edgecolor=Constants.color_black)

    # Show plot
    plt.show()