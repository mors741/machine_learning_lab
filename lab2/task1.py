import math
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator

GAUSS_COEF = 1 / math.sqrt(2 * math.pi)
PLOT_STEP = 0.5
MARGIN = 0.3


def min_max(list):
    min = list[0]
    max = list[0]
    for elem in list:
        if elem < min:
            min = elem
        elif elem > max:
            max = elem
    return min, max


def box_kernel(u):
    if 1 > u > -1:
        return 0.5
    else:
        return 0


def gaussian_kernel(u):
    return GAUSS_COEF * math.exp(-u ** 2 / 2)


def epanechnikov_kernel(u):
    if 1 > u > -1:
        return 0.75 * (1 - u ** 2)
    else:
        return 0


def triangular_kernel(u):
    u_abs = math.fabs(u)
    if u_abs < 1:
        return 1 - u_abs
    else:
        return 0


def interquartile_range(list):
    sorted_list = sorted(list)
    l = len(list)
    return sorted_list[l * 3 / 4] - sorted_list[l / 4]


def silverman_bandwidth(list):
    return 0.9 * min(np.std(list), interquartile_range(list) / 1.34) * math.pow(len(list), -0.2)


def calc_z_matrix(X, Y, x, y, kernel_function, hx, hy):
    shape = np.shape(X)
    Z = np.zeros(shape)
    for r in range(shape[0]):
        for c in range(shape[1]):
            val = 0
            for i in range(len(x)):
                val += kernel_function((X[r][c] - x[i]) / hx) * kernel_function((Y[r][c] - y[i]) / hy)
            Z[r][c] = val / (len(x) * hx * hy)
    return Z


def calc_volume(Z, cell_area):
    volume = 0
    for row in Z:
        for cell in row:
            volume += cell * cell_area
    return volume


def get_sample_dots(x1_list, x2_list):
    common_min_max = min_max(x1_list + x2_list)

    x_range = np.arange(common_min_max[0] - MARGIN, common_min_max[1] + MARGIN, PLOT_STEP)
    return np.meshgrid(x_range, x_range)


def surface_plot(X, Y, Z, title, z_limit=(0, 0), z_label="Density", power_limits=False):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    cb = fig.colorbar(surf, shrink=0.5, aspect=5)

    if power_limits:
        cb.formatter.set_powerlimits((0, 0))
        cb.update_ticks()
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

    if z_limit != (0,0):
        ax.set_zlim3d(z_limit[0], z_limit[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel(z_label)
    plt.title(title)


def contour_plot(X, Y, Z, title):
    fig = plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)

    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.xlabel("x1")
    plt.ylabel("x2")

    plt.title(title)


def draw_plots(X, Y, Z, title, z_limit=(0, 0)):
    surface_plot(X, Y, Z, title, z_limit)
    contour_plot(X, Y, Z, title)


def run(x1_list, x2_list):
    X, Y = get_sample_dots(x1_list, x2_list)

    Z_small = calc_z_matrix(X, Y, x1_list, x2_list, box_kernel, 0.15, 0.15)
    draw_plots(X, Y, Z_small, "Small bandwidth")

    silv_x = silverman_bandwidth(x1_list)
    silv_y = silverman_bandwidth(x2_list)
    Z_silv = calc_z_matrix(X, Y, x1_list, x2_list, box_kernel, silv_x, silv_y)
    draw_plots(X, Y, Z_silv, "Silverman's bandwidth")

    Z_big = calc_z_matrix(X, Y, x1_list, x2_list, box_kernel, 5, 5)
    draw_plots(X, Y, Z_big, "Big bandwidth")

    plt.show()
