from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

from lab2.task1 import get_sample_dots, draw_plots, silverman_bandwidth, gaussian_kernel, calc_z_matrix, box_kernel


def generate_indexes(sample_size, required_amount):
    res = []
    half_size = sample_size / 2
    indexes = range(0, sample_size)
    for i in range(required_amount):
        res.append(random.sample(indexes, half_size))
    return res

def calc_z_matrix_with_cache(X, Y, x, y, kernel_function, hx, hy):
    shape = np.shape(X)
    x_len = len(x)
    coef = hx * hy # should be later divided by x_len
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
    return sum_z / (l * len(index_matrix[0]))

def run(x1_list, x2_list):
    X, Y = get_sample_dots(x1_list, x2_list)

    silv_x = silverman_bandwidth(x1_list)
    silv_y = silverman_bandwidth(x2_list)
    cached_Z_silv = calc_z_matrix_with_cache(X, Y, x1_list, x2_list, gaussian_kernel, silv_x, silv_y)
    Z1 = calc_z_matrix(X, Y, x1_list, x2_list, gaussian_kernel, silv_x, silv_y)
    # Z2 = fold_z_matrix(cached_Z_silv, range(len(x1_list)/2, len(x1_list)))
    Z2 = variance(cached_Z_silv, generate_indexes(len(x1_list), 10))
    draw_plots(X, Y, Z1, "Z1")
    draw_plots(X, Y, Z2, "Z2")

    plt.show()
