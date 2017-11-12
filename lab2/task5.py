import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from lab2.task1 import get_sample_dots, calc_z_matrix, silverman_bandwidth, box_kernel, draw_plots, gaussian_kernel, \
    epanechnikov_kernel, triangular_kernel, calc_volume
from lab2.task3_bias import real_destribution
from lab2.task3_var import calc_Z_variance
from lab2.task4_var import get_silv_vector, lambda_h_list
from matplotlib.ticker import AutoMinorLocator


def calc_MISE(Z_real, X, Y, x1_list, x2_list, kernel_function, hx, hy):
    Z_restored = calc_z_matrix(X, Y, x1_list, x2_list, kernel_function, hx, hy)
    Z_bias_squared = np.square(Z_restored - Z_real)

    Z_variance = calc_Z_variance(X, Y, x1_list, x2_list, hx, hy, kernel_function=kernel_function)
    print "MISE calculated for hx=", hx
    return calc_volume(Z_bias_squared) + calc_volume(Z_variance)


def MISE_from_lambda(Z_real, X, Y, x1_list, x2_list, kernel_function):
    x_list = []
    y_list = []
    for l_h in lambda_h_list(x1_list, x2_list, start=0.15, stop=7.01, step=0.25):
        x_list.append(l_h[0])
        var = calc_MISE(Z_real, X, Y, x1_list, x2_list, kernel_function, l_h[1], l_h[2])
        y_list.append(var)
    return x_list, y_list


def visualize_MISE_different_kernels(Z_real, X, Y, x1_list, x2_list):
    plt.figure()
    x_list1, y_list1 = MISE_from_lambda(Z_real, X, Y, x1_list, x2_list, box_kernel)
    x_list2, y_list2 = MISE_from_lambda(Z_real, X, Y, x1_list, x2_list, gaussian_kernel)
    x_list3, y_list3 = MISE_from_lambda(Z_real, X, Y, x1_list, x2_list, epanechnikov_kernel)
    x_list4, y_list4 = MISE_from_lambda(Z_real, X, Y, x1_list, x2_list, triangular_kernel)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x_list1, y_list1, label="Box kernel", marker='.')
    plt.plot(x_list2, y_list2, label="Gaussian kernel", marker='.')
    plt.plot(x_list3, y_list3, label="Epanechnikov kernel", marker='.')
    plt.plot(x_list4, y_list4, label="Triangular kernel", marker='.')
    plt.xlabel("lambda (h/h_silverman)")
    plt.ylabel("MISE")
    plt.yscale('log')
    plt.title("MISE dependency from lambda (different kernels)")
    plt.legend()


def run(x1_list, x2_list):
    X, Y = get_sample_dots(x1_list, x2_list)
    Z_real = real_destribution(X, Y)

    visualize_MISE_different_kernels(Z_real, X, Y, x1_list, x2_list)



    plt.show()
