from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from lab2.task1 import silverman_bandwidth, gaussian_kernel, box_kernel, epanechnikov_kernel, triangular_kernel
from lab2.task3_var import generate_indexes
from matplotlib.ticker import AutoMinorLocator

ITER_COUNT = 100
FIXED_POINT_1 = (0.0, 4.0)
FIXED_POINT_2 = (6.0, 6.0)
FIXED_POINT_3 = (0.5, 3.0)


def calc_fixed_point_with_cache(x_fixed, y_fixed, x_list, y_list, kernel_function, hx, hy, expected_sample_size):
    x_len = len(x_list)
    values = np.zeros(x_len)
    coef = expected_sample_size * hx * hy
    for i in range(x_len):
        values[i] = (kernel_function((x_fixed - x_list[i]) / hx) * kernel_function((y_fixed - y_list[i]) / hy)) / coef
    return values


def fold_fixed_point(cached_fixed_point, indexes):
    val = 0
    for i in indexes:
        val += cached_fixed_point[i]
    return val

def variance(cached_fixed_point, index_matrix):
    folded_z_storage = []
    sum_z = 0
    l = len(index_matrix)
    for i in range(l):
        z_value = fold_fixed_point(cached_fixed_point, index_matrix[i])
        folded_z_storage.append(z_value)
        sum_z += z_value
    mean = sum_z / l
    var = 0
    for z_iter in folded_z_storage:
        var += (z_iter - mean)**2
    var = var / l
    return var


def calc_fixed_point_variance(x_fixed, y_fixed, x_list, y_list, kernel_function, hx, hy):
    cached_fixed_point = calc_fixed_point_with_cache(x_fixed, y_fixed, x_list, y_list, kernel_function, hx, hy, len(x_list) / 2)
    return variance(cached_fixed_point, generate_indexes(len(x_list), ITER_COUNT))

def visualize_fixed_points(x1, x2):
    plt.figure()
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.scatter(x1, x2, marker=".")
    plt.scatter(*FIXED_POINT_1, marker="o", label="fixed point 1")
    plt.scatter(*FIXED_POINT_2, marker="o", label="fixed point 2")
    plt.scatter(*FIXED_POINT_3, marker="o", label="fixed point 3")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Fixed points")
    plt.legend()


def get_silv_vector(x1_list, x2_list):
    silv_x = silverman_bandwidth(x1_list)
    silv_y = silverman_bandwidth(x2_list)
    return (silv_x, silv_y)


def zip_silv_lambda(silv_vector, lmb_list):
    zipped = []
    for lmb in lmb_list:
        zipped.append((lmb, silv_vector[0]*lmb, silv_vector[1]*lmb))
    return zipped


def lambda_h_list(x1_list, x2_list, start=0.25, stop=7.01, step=0.25):
    lmb_list = np.arange(start, stop, step)
    # lmb_list = np.logspace(1, 2.6, 10)/50
    silv_vector = get_silv_vector(x1_list, x2_list)
    return zip_silv_lambda(silv_vector, lmb_list)


def variance_from_lambda(fixed_point, x1_list, x2_list, kernel_function):
    x_list = []
    y_list = []
    for l_h in lambda_h_list(x1_list, x2_list):
        x_list.append(l_h[0])
        var = calc_fixed_point_variance(fixed_point[0], fixed_point[1], x1_list, x2_list,
                                        kernel_function, l_h[1], l_h[2])
        y_list.append(var)
    return x_list, y_list


def visualize_different_points(x1_list, x2_list, kernel_function):
    plt.figure()
    x_list1, y_list1 = variance_from_lambda(FIXED_POINT_1, x1_list, x2_list, kernel_function)
    x_list2, y_list2 = variance_from_lambda(FIXED_POINT_2, x1_list, x2_list, kernel_function)
    x_list3, y_list3 = variance_from_lambda(FIXED_POINT_3, x1_list, x2_list, kernel_function)
    ax = plt.axes()
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x_list1, y_list1, label="fixed point 1", marker='.', color="C1")
    plt.plot(x_list2, y_list2, label="fixed point 2", marker='.', color="C2")
    plt.plot(x_list3, y_list3, label="fixed point 3", marker='.', color="C3")
    plt.xlabel("lambda (h/h_silverman)")
    plt.ylabel("Variance")
    plt.yscale('log')
    plt.title("Variance dependency from lambda (different points, Gaussian kernel)")
    plt.legend()

def visualize_different_kernels(x1_list, x2_list, fixed_point):
    plt.figure()
    x_list1, y_list1 = variance_from_lambda(fixed_point, x1_list, x2_list, box_kernel)
    x_list2, y_list2 = variance_from_lambda(fixed_point, x1_list, x2_list, gaussian_kernel)
    x_list3, y_list3 = variance_from_lambda(fixed_point, x1_list, x2_list, epanechnikov_kernel)
    x_list4, y_list4 = variance_from_lambda(fixed_point, x1_list, x2_list, triangular_kernel)
    ax = plt.axes()
    # ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(x_list1, y_list1, label="Box kernel", marker='.')
    plt.plot(x_list2, y_list2, label="Gaussian kernel", marker='.')
    plt.plot(x_list3, y_list3, label="Epanechnikov kernel", marker='.')
    plt.plot(x_list4, y_list4, label="Triangular kernel", marker='.')
    plt.xlabel("lambda (h/h_silverman)")
    plt.ylabel("Variance")
    plt.yscale('log')
    plt.title("Variance dependency from lambda (different kernels)")
    plt.legend()

def run(x1_list, x2_list):
    visualize_fixed_points(x1_list, x2_list)
    visualize_different_points(x1_list, x2_list, gaussian_kernel)
    visualize_different_kernels(x1_list, x2_list, FIXED_POINT_1)

    plt.show()


