from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal

from lab2.task1 import gaussian_kernel, box_kernel, epanechnikov_kernel, triangular_kernel
from matplotlib.ticker import AutoMinorLocator

from lab2.task4_var import FIXED_POINT_1, FIXED_POINT_2, FIXED_POINT_3, lambda_h_list


def calc_fixed_point_real_dens(fixed_point):
    return 0.2 * multivariate_normal.pdf(fixed_point, [0, 0], [[1, 0], [0, 1]]) + \
           0.5 * multivariate_normal.pdf(fixed_point, [0, 4], [[2, 0.6], [0.6, 0.5]]) + \
           0.3 * multivariate_normal.pdf(fixed_point, [5, 7], [[2, -0.8], [-0.8, 2]])


def calc_fixed_point_restored_dens(x_fixed, y_fixed, x_list, y_list, kernel_function, hx, hy):
    x_len = len(x_list)
    dens = 0
    for i in range(x_len):
        dens += (kernel_function((x_fixed - x_list[i]) / hx) * kernel_function((y_fixed - y_list[i]) / hy))
    return dens / (x_len * hx * hy)


def calc_fixed_point_bias(fixed_point, x_list, y_list, kernel_function, hx, hy):
    real_dens = calc_fixed_point_real_dens(fixed_point)
    restored_dens = calc_fixed_point_restored_dens(fixed_point[0], fixed_point[1], x_list, y_list, kernel_function, hx, hy)
    return math.fabs(restored_dens - real_dens)


def bias_from_lambda(fixed_point, x1_list, x2_list, kernel_function):
    x_list = []
    y_list = []
    for l_h in lambda_h_list(x1_list, x2_list, start=0.15, stop=8.01, step=0.1):
        x_list.append(l_h[0])
        var = calc_fixed_point_bias(fixed_point, x1_list, x2_list, kernel_function, l_h[1], l_h[2])
        y_list.append(var)
    return x_list, y_list

def visualize_different_points(x1_list, x2_list, kernel_function):
    plt.figure()
    x_list1, y_list1 = bias_from_lambda(FIXED_POINT_1, x1_list, x2_list, kernel_function)
    x_list2, y_list2 = bias_from_lambda(FIXED_POINT_2, x1_list, x2_list, kernel_function)
    x_list3, y_list3 = bias_from_lambda(FIXED_POINT_3, x1_list, x2_list, kernel_function)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x_list1, y_list1, label="fixed point 1", marker='.', color="C1")
    plt.plot(x_list2, y_list2, label="fixed point 2", marker='.', color="C2")
    plt.plot(x_list3, y_list3, label="fixed point 3", marker='.', color="C3")
    plt.xlabel("lambda (h/h_silverman)")
    plt.ylabel("Bias")
    plt.title("Bias dependency from lambda (different points)")
    plt.legend()


def visualize_different_kernels(x1_list, x2_list, fixed_point):
    plt.figure()
    x_list1, y_list1 = bias_from_lambda(fixed_point, x1_list, x2_list, box_kernel)
    x_list2, y_list2 = bias_from_lambda(fixed_point, x1_list, x2_list, gaussian_kernel)
    x_list3, y_list3 = bias_from_lambda(fixed_point, x1_list, x2_list, epanechnikov_kernel)
    x_list4, y_list4 = bias_from_lambda(fixed_point, x1_list, x2_list, triangular_kernel)
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
    plt.ylabel("Bias")
    plt.title("Bias dependency from lambda (different kernels)")
    plt.legend()


def run(x1_list, x2_list):

    visualize_different_points(x1_list, x2_list, box_kernel)
    visualize_different_points(x1_list, x2_list, gaussian_kernel)
    visualize_different_points(x1_list, x2_list, epanechnikov_kernel)
    visualize_different_points(x1_list, x2_list, triangular_kernel)

    visualize_different_kernels(x1_list, x2_list, FIXED_POINT_1)
    visualize_different_kernels(x1_list, x2_list, FIXED_POINT_2)
    visualize_different_kernels(x1_list, x2_list, FIXED_POINT_3)

    plt.show()


