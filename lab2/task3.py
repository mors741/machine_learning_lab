from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

from lab2.task1 import get_sample_dots, draw_plots, silverman_bandwidth, gaussian_kernel, calc_z_matrix, box_kernel


def normal_distribution(X, Y, mean, sigma):
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    rv = multivariate_normal(mean, sigma)
    return rv.pdf(pos)

def real_destribution(X, Y):
    Z1 = 0.2 * normal_distribution(X, Y, [0, 0], [[1, 0], [0, 1]])
    Z2 = 0.5 * normal_distribution(X, Y, [0, 4], [[2, 0.6], [0.6, 0.5]])
    Z3 = 0.3 * normal_distribution(X, Y, [5, 7], [[2, -0.8], [-0.8, 2]])
    return Z1 + Z2 + Z3


def run(x1_list, x2_list):
    X, Y = get_sample_dots(x1_list, x2_list)
    Z_real = real_destribution(X, Y)

    draw_plots(X, Y, Z_real, "Real distribution density")

    silv_x = silverman_bandwidth(x1_list)
    silv_y = silverman_bandwidth(x2_list)
    Z_silv = calc_z_matrix(X, Y, x1_list, x2_list, box_kernel, silv_x, silv_y)
    draw_plots(X, Y, Z_silv, "Silverman's bandwidth")

    draw_plots(X, Y, np.absolute(Z_silv - Z_real), "Bias")



    plt.show()
