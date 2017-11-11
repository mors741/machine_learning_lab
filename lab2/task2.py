from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

from lab2.task1 import get_sample_dots, calc_z_matrix, silverman_bandwidth, box_kernel, draw_plots, gaussian_kernel, \
    epanechnikov_kernel, triangular_kernel


def run(x1_list, x2_list):
    X, Y = get_sample_dots(x1_list, x2_list)
    silv_x = silverman_bandwidth(x1_list)
    silv_y = silverman_bandwidth(x2_list)

    Z = calc_z_matrix(X, Y, x1_list, x2_list, box_kernel, silv_x, silv_y)
    draw_plots(X, Y, Z, "Box kernel")

    Z = calc_z_matrix(X, Y, x1_list, x2_list, gaussian_kernel, silv_x, silv_y)
    draw_plots(X, Y, Z, "Gaussian kernel")

    Z = calc_z_matrix(X, Y, x1_list, x2_list, epanechnikov_kernel, silv_x, silv_y)
    draw_plots(X, Y, Z, "Epanechnikov kernel")

    Z = calc_z_matrix(X, Y, x1_list, x2_list, triangular_kernel, silv_x, silv_y)
    draw_plots(X, Y, Z, "Triangular kernel")

    plt.show()
