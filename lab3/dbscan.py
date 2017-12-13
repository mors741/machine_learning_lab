import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabaz_score, silhouette_score

from lab3.visual import visualize


def count_clusters_and_noise(labels):
    clusters = set()
    noise = 0
    for label in labels:
        if label == -1:
            noise += 1
        else:
            clusters.add(label)
    return len(clusters), noise


def dbscan_params(eps, m_s, data):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    k, noise = count_clusters_and_noise(db.labels_)
    if k > 1:
        chi = calinski_harabaz_score(data, db.labels_)
        si = silhouette_score(data, db.labels_)
    else:
        chi = 1
        si = -1
    return k, noise, chi, si


def image_plot(X, Y, Z, title, z_label, max_chi=None, max_si=None, cmap=cm.jet, show_max=True):
    plt.figure()
    plt.imshow(Z, extent=[X[0][0], X[0][-1], Y[0][0], Y[-1][0]], cmap=cmap, aspect="auto", origin='lower',
               interpolation="none")

    cb = plt.colorbar()
    cb.set_label(z_label)

    if show_max:
        if max_chi is not None:
            i, j = max_chi[1:]
            plt.scatter(X[i][j], Y[i][j], marker="o", s=80, facecolors='none', edgecolors='w', label="Max CHI")

        if max_si is not None:
            i, j = max_si[1:]
            plt.scatter(X[i][j], Y[i][j], marker=",", s=80, facecolors='none', edgecolors='w', label="Max SI")
        plt.legend(loc=4)

    plt.xlabel("eps")
    plt.ylabel("m_s")
    plt.title(title)


def print_max(name, max_val, E, M, K, N, CHI, SI):
    i, j = max_val[1:]
    print name + ": [eps=" + str(E[i][j]) + ", m_s=" + str(M[i][j]) + ", K=" + str(K[i][j]) \
          + ", noise=" + str(N[i][j]) + ", CHI=" + str(CHI[i][j]) + ", SI=" + str(SI[i][j]) + "]"


def run(data, eps_range, ms_range, cmap=cm.jet, show_max=True):
    E, M = np.meshgrid(eps_range, ms_range)
    shape = np.shape(E)
    K = np.empty(shape)
    N = np.empty(shape)
    CHI = np.empty(shape)
    SI = np.empty(shape)
    max_chi = (0, -1, -1)  # (chi, eps, m_s)
    max_si = (-1, -1, -1)  # (si, eps, m_s)
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
    image_plot(E, M, K, "Clusters", "K", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, N, "Noise", "Noise", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, CHI, "Calinski-Harabasz index", "CHI", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, SI, "Silhouette Coefficient", "SI", max_chi, max_si, cmap=cmap, show_max=show_max)

    print "--"
    print_max("max_chi", max_chi, E, M, K, N, CHI, SI)
    print_max("max_si", max_si, E, M, K, N, CHI, SI)

    best_chi_labels = DBSCAN(eps=E[max_chi[1]][max_chi[2]], min_samples=M[max_chi[1]][max_chi[2]]).fit(data).labels_
    best_si_labels = DBSCAN(eps=E[max_si[1]][max_si[2]], min_samples=M[max_si[1]][max_si[2]]).fit(data).labels_

    if np.shape(data)[1] == 2:
        visualize(data, best_chi_labels, "Max CHI (" + "{:.1f}".format(CHI[max_chi[1]][max_chi[2]]) + ")")
        visualize(data, best_si_labels, "Max SI (" + "{:.3f}".format(SI[max_si[1]][max_si[2]]) + ")")
    else:
        print "Couldn't visualize clusterization result. Dimensions != 2"

    return best_chi_labels, best_si_labels
