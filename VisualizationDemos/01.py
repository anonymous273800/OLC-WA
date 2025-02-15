import numpy as np
import matplotlib.pyplot as plt

def linear_regression(xs, ys):
    x0 = np.ones(len(xs))
    xs = np.concatenate((np.matrix(x0).T, xs), axis=1)

    w = _pseudo_inverse_linear_regression(xs, ys)
    w = np.asarray(w).reshape(-1)  # convert to one 1 array
    return w

def _pseudo_inverse_linear_regression(X, y):
    XT = np.transpose(X)
    x_pseudo_inv = np.matmul(np.linalg.inv(np.matmul(XT, X)), XT)
    w = np.matmul(x_pseudo_inv, y)
    return w


def linear_regression(xs, ys):
    x0 = np.ones(len(xs))
    xs = np.concatenate((np.matrix(x0).T, xs), axis=1)

    w = _pseudo_inverse_linear_regression(xs, ys)
    w = np.asarray(w).reshape(-1)  # convert to one 1 array
    return w

def compute_predictions_(xs, w, b):
    y_predicted = np.array([(np.dot(x, w) + b) for x in xs]).flatten()
    return y_predicted

def _compute_predictions_(xs, w):
    b = w[0]
    w = w[1:]
    return compute_predictions_(xs, w, b)

def _compute_predictions_(xs, w):
    b = w[0]
    w = w[1:]
    return compute_predictions_(xs, w, b)


def predict(w, x):
    slope, bias = w

    # Calculate y using linear regression equation: y = slope * x + bias
    y = slope * x + bias
    return y

def generate_data_using_lines_and_compute_batch_regression():
    x_values1 = np.linspace(-15, 15, 100)
    y_values_line1 = line1(x_values1)

    x_values2 = np.linspace(-15, 15, 100)
    y_values_line2 = line2(x_values2)

    X = np.concatenate((x_values1, x_values2))
    y = np.concatenate((y_values_line1, y_values_line2))
    X = X.reshape(-1,1)
    w = linear_regression(X,y)
    y_pred = _compute_predictions_(X, w)
    # plt.plot(X, y_pred, color='orange', label='Line Using Batch Regression')


def define_plane_from_norm_vector_and_a_point(nv, point):
    temp = -1 * sum(n * v for n, v in zip(nv, point))
    nv = np.append(nv, temp)
    return nv

def find_intersection():
    # Solve for x
    x = (-1 - 2) / (3 - .5)  # (c2 - c1) / (m1 - m2)
    # Compute y using x in either line equation
    y = line1(x)  # Using line1
    # Return the intersection point
    return x, y

def line_from_plane(plane, x_values):
    a, b, c = plane[:3]
    return (-c - a * x_values) / b

def line1(x):
    return 3*x + 2

def line2(x):
    return .5*x -1

def avg_line(plane, x):
    # Extracting coefficients from the plane
    a, b= plane[:-1]
    d = plane[-1]

    # Calculating the y value using the equation of the line
    y = (-a / b) * x - (d / b)

    return y

def expr1():
    x_values = np.linspace(-10, 10, 100)
    y_values_line1 = line1(x_values)
    x_values2 = np.linspace(-30, 30, 100)
    y_values_line2 = line2(x_values2)

    # Plot the lines
    plt.plot(x_values2, y_values_line2, label='Base Decision Boundary y = .5x -1', color='maroon')
    plt.plot(x_values, y_values_line1, label='Incremental Decision Boundary y = 3x + 2', color='blue')



    plt.text(-2, line1(-2) -.3 , '$\mathscr{y = 3x + 2}$' , fontsize=10, color='blue', rotation=73)
    plt.text(-4, line2(-4) + .23, '$\mathscr{y = .5x -1}$', fontsize=10, color='maroon', rotation=25)

    # Plot the intersection point
    x_intersect, y_intersect = find_intersection()
    plt.scatter(x_intersect, y_intersect, color=(0.2, .3, .3),
                label=f'Intersection Point ({x_intersect:.2f}, {y_intersect:.2f})')

    # Plot the normal vectors
    norm_v1 = np.array([3, -1])
    norm_v2 = np.array([.5, -1])

    norm_v1 = norm_v1 / np.sqrt( (norm_v1 * norm_v1).sum() )
    norm_v2 = norm_v2 / np.sqrt((norm_v2 * norm_v2).sum())

    plt.quiver(.5, line1(.5), norm_v1[0], norm_v1[1], angles='xy', scale_units='xy', scale=1, color='blue',
               label='Normal Vector - Incremental Decision Boundary', width=0.003)
    plt.text(.5 + 1, line1(.5) - .3, '$\mathscr{n1}$', fontsize=10, color='blue')
    plt.quiver(5, line2(5), norm_v2[0], norm_v2[1], angles='xy', scale_units='xy', scale=1, color='maroon',
               label='Normal Vector - Base Decision Boundary', width=0.003)
    plt.text(5 +.5, line2(5) - 1, '$\mathscr{n2}$', fontsize=10, color='maroon')

    # # Calculate the sum of the two norm vectors
    # print("norm_v1 , norm_v2 ",norm_v1 , norm_v2)
    # norm_v1 = .5 * norm_v1
    # norm_v2 = .5 * norm_v2
    # sum_vector = norm_v1 +  norm_v2
    # print("sum_vector", sum_vector)

    # # Plot the first norm vector
    # plt.quiver(0, 0, .5*norm_v1[0], .5*norm_v1[1], angles='xy', scale_units='xy', scale=1, color='blue',
    #            label='_nolegend_', width=0.003)

    # # Plot the second norm vector starting from the end point of the first vector
    # plt.quiver(.5*norm_v1[0], .5*norm_v1[1], .5*norm_v2[0], .5* norm_v2[1], angles='xy', scale_units='xy', scale=1, color='maroon',
    #            label='_nolegend_', width=0.003)

    x_intersect, y_intersect = find_intersection()
    intersection_point = [x_intersect, y_intersect]

    # CASE1: Calculate the average normal vector
    w_base = 0.5
    w_inc = 0.5
    avg_norm_v = ((norm_v1 * w_base + norm_v2 * w_inc)) / (w_base + w_inc)

    plane = define_plane_from_norm_vector_and_a_point(avg_norm_v, intersection_point)

    # Plot the average normal vector
    # plt.quiver(2, avg_line(plane, 2), avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1,
    #            color='green', label='Weighted Average Normal Vector', width=0.003)
    plt.quiver(2, avg_line(plane, 2), avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1,
               color='green', label='_', width=0.003)

    # plt.text(2 + .7, avg_line(plane, 2) - .8, '$V_{\mathrm{avg}}$', fontsize=6, color='green')

    # Plot the average normal vector
    plt.quiver(2, avg_line(plane, 2), avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1,
               color='green', label='Weighted Average Normal Vectors', width=0.003)



    # Plot the line defined by the plane coefficients
    y_values_plane = line_from_plane(plane, x_values)
    y_values_plane = y_values_plane  # + 0.2  # this is just adjusting y a little bit to allow both of the lines to appear in the plot
    # plt.plot(x_values, y_values_plane, color='green', label='OLR-WA Method $W_{\mathrm{base}}=.5$, $W_{\mathrm{inc}}=.5$')
    plt.plot(x_values, y_values_plane, color='green',
             label='$\mathscr{l}_{1}:$ OLC-WA, $\\alpha=.5$')
    plt.text(3 - .4, avg_line(plane, 3), '$\mathscr{l}_{1}$', fontsize=12, color='darkgreen',
             rotation=75, fontdict={'weight': 'bold'})

    # CASE2: Calculate the average normal vector
    w_base = 0.8
    w_inc = 0.2
    avg_norm_v = ((norm_v1 * w_base + norm_v2 * w_inc)) / (w_base + w_inc)

    plane = define_plane_from_norm_vector_and_a_point(avg_norm_v, intersection_point)

    # Plot the average normal vector
    # plt.quiver(2, avg_line(plane, 2), avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1,
    #            color='green', label='Weighted Average Normal Vector', width=0.003)
    plt.quiver(2, avg_line(plane, 2), avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1,
               color='green', label='_', width=0.003)

    # plt.text(1.3 + 1, avg_line(plane, 1.3) - .8, '$V_{\mathrm{avg}}$', fontsize=6, color='green')

    # Plot the line defined by the plane coefficients
    y_values_plane = line_from_plane(plane, x_values)
    y_values_plane = y_values_plane  # + 0.2  # this is just adjusting y a little bit to allow both of the lines to appear in the plot
    # plt.plot(x_values, y_values_plane, color='green', label='OLR-WA Method $W_{\mathrm{base}}=.8$, $W_{\mathrm{inc}}=.2$')
    plt.plot(x_values, y_values_plane, color='green',
             label='$\mathscr{l}_{2}:$ OLC-WA, $\\alpha=.8$')
    plt.text(.8-.35, avg_line(plane, .8), r'$\mathscr{l}_{2}$', fontsize=12, color='darkgreen',
             rotation=75)

    # CASE3: Calculate the average normal vector
    w_base = 0.2
    w_inc = 0.8
    avg_norm_v = ((norm_v1 * w_base + norm_v2 * w_inc)) / (w_base + w_inc)

    plane = define_plane_from_norm_vector_and_a_point(avg_norm_v, intersection_point)



    # plt.text(2 + .7, avg_line(plane, 2) - .8, '$V_{\mathrm{avg}}$', fontsize=8, color='green')

    # Plot the line defined by the plane coefficients
    y_values_plane = line_from_plane(plane, x_values)
    y_values_plane = y_values_plane  # + 0.2  # this is just adjusting y a little bit to allow both of the lines to appear in the plot
    # plt.plot(x_values, y_values_plane, color='green', label='OLR-WA Method $W_{\mathrm{base}}=.2$, $W_{\mathrm{inc}}=.8$')
    plt.plot(x_values, y_values_plane, color='green',
             label='$\mathscr{l}_{3}:$ OLC-WA, $\\alpha=.2$')
    plt.text(4, avg_line(plane,4 )+.3 , '$\mathscr{l}_{3}$', fontsize=12, color='darkgreen', rotation=35, fontweight='bold')

    plt.quiver(2, avg_line(plane, 2), avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1,
               color='green', label='_', width=0.003)



    #batch regression
    generate_data_using_lines_and_compute_batch_regression()

    #####################################################

    # Plot the first norm vector
    plt.quiver(0, 0, .5 * norm_v1[0], .5 * norm_v1[1], angles='xy', scale_units='xy', scale=1, color='blue',
               label='_nolegend_', width=0.003)

    # Plot the second norm vector starting from the end point of the first vector
    plt.quiver(.5 * norm_v1[0], .5 * norm_v1[1], .5 * norm_v2[0], .5 * norm_v2[1], angles='xy', scale_units='xy',
               scale=1, color='maroon',
               label='_nolegend_', width=0.003)

    # Calculate the average normal vector
    w_base = 0.5
    w_inc = 0.5
    print("norm_v1 , norm_v2 ", norm_v1, norm_v2)
    avg_norm_v = ((norm_v1 * w_base + norm_v2 * w_inc)) / (w_base + w_inc)
    print("avg_norm_v", avg_norm_v)

    # Plot the sum vector
    plt.quiver(0, 0, avg_norm_v[0], avg_norm_v[1], angles='xy', scale_units='xy', scale=1, color='green',
               width=0.003, label='_nolegend_')
    #####################################################


    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    plt.axis('equal')  # Set equal scaling for x and y axes
    plt.rcParams['legend.fontsize'] = 7.8
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    expr1()
