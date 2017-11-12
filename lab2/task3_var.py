from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

from lab2.task1 import get_sample_dots, draw_plots, silverman_bandwidth, gaussian_kernel, calc_z_matrix, box_kernel, \
    surface_plot

ITER_COUNT = 10


def generate_indexes(sample_size, required_amount):
    res = []
    half_size = sample_size / 2
    indexes = range(0, sample_size)
    for i in range(required_amount):
        res.append(random.sample(indexes, half_size))
    return res

def calc_z_matrix_with_cache(X, Y, x, y, kernel_function, hx, hy, expected_sample_size):
    shape = np.shape(X)
    x_len = len(x)
    coef = expected_sample_size * hx * hy
    Z = np.zeros((shape[0], shape[1], x_len))

    for r in range(shape[0]):
        for c in range(shape[1]):
            for i in range(x_len):
                Z[r][c][i] = (kernel_function((X[r][c] - x[i]) / hx) * kernel_function((Y[r][c] - y[i]) / hy)) / coef
    return Z


def fold_z_matrix(cached_Z, indexes):
    shape = np.shape(cached_Z)
    res_Z = np.zeros((shape[0], shape[1]))
    for r in range(shape[0]):
        for c in range(shape[1]):
            for i in indexes:
                res_Z[r][c] += cached_Z[r][c][i]
    return res_Z

def variance(cached_Z, index_matrix):
    folded_z_storage = []
    shape = np.shape(cached_Z)
    sum_z = np.zeros((shape[0], shape[1]))
    l = len(index_matrix)
    for i in range(l):
        Z = fold_z_matrix(cached_Z, index_matrix[i])
        folded_z_storage.append(Z)
        sum_z = np.add(sum_z, Z)
    mean_matrix = sum_z / l
    variance_matrix = np.zeros((shape[0], shape[1]))
    for z_iter in folded_z_storage:
        for c in range(shape[0]):
            for r in range(shape[1]):
                variance_matrix[c][r] += (z_iter[c][r] - mean_matrix[c][r])**2
    variance_matrix = variance_matrix / l
    return variance_matrix


def calc_Z_variance(X, Y, x, y, hx, hy, kernel_function=gaussian_kernel):
    cached_Z = calc_z_matrix_with_cache(X, Y, x, y, kernel_function, hx, hy, len(x) / 2)
    return variance(cached_Z, generate_indexes(len(x), ITER_COUNT))


def run(x1_list, x2_list):
    X, Y = get_sample_dots(x1_list, x2_list)

    Z = calc_Z_variance(X, Y, x1_list, x2_list, 0.15, 0.15)
    surface_plot(X, Y, Z, "Variance (small bandwidth)", z_label="Variance", power_limits=True)

    silv_x = silverman_bandwidth(x1_list)
    silv_y = silverman_bandwidth(x2_list)
    Z = calc_Z_variance(X, Y, x1_list, x2_list, silv_x, silv_y)
    surface_plot(X, Y, Z, "Variance (Silverman's bandwidth)", z_label="Variance", power_limits=True)

    Z = calc_Z_variance(X, Y, x1_list, x2_list, 5, 5)
    surface_plot(X, Y, Z, "Variance (big bandwidth)", z_label="Variance", power_limits=True)

    plt.show()
