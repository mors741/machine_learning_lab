import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabaz_score, silhouette_score

from lab3.visual import cluster_coordinates, visualize


def count_clusters_and_noise(x_clusters):
    size = len(x_clusters)
    if -1 in x_clusters:
        return size - 1, len(x_clusters[-1])
    else:
        return size, 0


def dbscan_params(eps, m_s, data):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    x_clusters, y_clusters = cluster_coordinates(data, db.labels_)
    k, noise = count_clusters_and_noise(x_clusters)
    # print "(eps, m_s) =", (eps, m_s)
    if k > 1:
        chi = calinski_harabaz_score(data, db.labels_)
        si = silhouette_score(data, db.labels_)
    else:
        chi = 1
        si = -1
    return k, noise, chi, si


def dbscan_visualize(eps, m_s, data, title):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    visualize(data, db.labels_, title)


def surface_plot(X, Y, Z, title, z_label, z_limit=(0, 0)):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    fig.colorbar(surf, shrink=0.5, aspect=5)

    if z_limit != (0, 0):
        ax.set_zlim3d(z_limit[0], z_limit[1])
    ax.set_xlabel("eps")
    ax.set_ylabel("m_s")
    ax.set_zlabel(z_label)
    plt.title(title)


def image_plot(X, Y, Z, title, z_label, max_chi=None, max_si=None):
    plt.figure()
    plt.imshow(Z, extent=[X[0][0], X[0][-1], Y[0][0], Y[-1][0]], cmap=cm.jet, aspect="auto", origin='lower',
               interpolation="none")

    cb = plt.colorbar()
    cb.set_label(z_label)

    if max_chi is not None:
        i, j = max_chi[1:]
        plt.scatter(X[i][j], Y[i][j], marker="o", facecolors='none', edgecolors='w', label="Max CHI")

    if max_si is not None:
        i, j = max_si[1:]
        plt.scatter(X[i][j], Y[i][j], marker=",", facecolors='none', edgecolors='w', label="Max SI")

    plt.xlabel("eps")
    plt.ylabel("m_s")
    plt.title(title)


def print_max(name, max_val, E, M, K, N, CHI, SI):
    i, j = max_val[1:]
    print name+": [eps=" + str(E[i][j]) + ", m_s="+str(M[i][j])+ ", K="+ str(K[i][j])+", noise "+str(N[i][j])+", CHI="+str(CHI[i][j])+", SI="+str(SI[i][j])+"]"

def run(data):
    eps_list = np.arange(0.02, 0.25, 0.05)
    ms_list = xrange(6, 250, 20)
    # eps_list = np.arange(0.02, 0.25, 0.003)
    # ms_list = xrange(6, 250, 3)
    # eps_list = np.arange(0.02, 0.1, 0.01)
    # ms_list = xrange(2, 20, 5)
    E, M = np.meshgrid(eps_list, ms_list)
    shape = np.shape(E)
    K = np.empty(shape)
    N = np.empty(shape)
    CHI = np.empty(shape)
    SI = np.empty(shape)
    max_chi = (0, -1, -1) # (chi, eps, m_s)
    max_si = (-1, -1, -1) # (si, eps, m_s)
    for i in xrange(shape[0]):
        print str(M[i][0]) + "/" + str(M[-1][0])
        for j in xrange(shape[1]):
            k, noise, chi, si = dbscan_params(E[i][j], M[i][j], data)
            K[i][j] = k
            N[i][j] = noise
            CHI[i][j] = chi
            SI[i][j] = si
            if chi > max_chi[0]:
                max_chi = (chi, i, j)
            if si > max_si[0]:
                max_si = (si, i, j)
    image_plot(E, M, K, "K", "K", max_chi, max_si)
    image_plot(E, M, N, "N", "N", max_chi, max_si)
    image_plot(E, M, CHI, "CHI", "CHI", max_chi, max_si)
    image_plot(E, M, SI, "SI", "SI", max_chi, max_si)

    print_max("max_chi", max_chi, E, M, K, N, CHI, SI)
    print_max("max_si", max_si, E, M, K, N, CHI, SI)


    dbscan_visualize(E[max_chi[1]][max_chi[2]], M[max_chi[1]][max_chi[2]], data, "Max CHI ("+"{:.1f}".format(CHI[max_chi[1]][max_chi[2]])+")")
    dbscan_visualize(E[max_si[1]][max_si[2]], M[max_si[1]][max_si[2]], data, "Max SI ("+"{:.3f}".format(SI[max_si[1]][max_si[2]])+")")


    plt.show()
    # visualize_by_cluster_coordinates(x_clusters, y_clusters)
