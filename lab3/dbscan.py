import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabaz_score, silhouette_score, adjusted_rand_score

from lab3 import heat_map
from lab3.visual import visualize

def visualize_dbscan(eps, m_s, data, title):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    visualize(data, db.labels_, title)
    heat_map.run(data, db.labels_, title=title)

def count_clusters_and_noise(data, labels, true_labels):
    clusters = set()
    noise = 0
    clear_data = []
    clear_labels = []
    clear_true_labels = []
    for i in xrange(len(labels)):
        if labels[i] == -1:
            noise += 1
        else:
            clusters.add(labels[i])
            clear_data.append(data[i])
            clear_labels.append(labels[i])
            clear_true_labels.append(true_labels[i])
    return len(clusters), noise, clear_data, clear_labels, clear_true_labels


def dbscan_params(eps, m_s, data, true_labels):
    db = DBSCAN(eps=eps, min_samples=m_s).fit(data)
    k, noise, clear_data, clear_labels, clear_true_labels = count_clusters_and_noise(data, db.labels_, true_labels)
    if k > 1:
        chi = calinski_harabaz_score(clear_data, clear_labels)
        si = silhouette_score(clear_data, clear_labels)
    else:
        chi = 1
        si = -1
    ari = adjusted_rand_score(clear_true_labels, clear_labels)
    return k, noise, chi, si, ari


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
        plt.legend(loc=2)

    plt.xlabel("eps")
    plt.ylabel("m_s")
    plt.ylim(ymin=np.min(Y))
    plt.title(title)


def print_max(name, max_val, E, M, K, N, CHI, SI, ARI):
    i, j = max_val[1:]
    print name + ": [eps=" + str(E[i][j]) + ", m_s=" + str(M[i][j]) + ", K=" + str(K[i][j]) \
          + ", noise=" + str(N[i][j]) + ", CHI=" + str(CHI[i][j]) + ", SI=" + str(SI[i][j]) + ", ARI=" + str(ARI[i][j]) + "]"


def run(data, eps_range, ms_range, true_labels, cmap=cm.jet, show_max=True, noise_limit=1):
    sample_size = len(data)
    E, M = np.meshgrid(eps_range, ms_range)
    shape = np.shape(E)
    K = np.empty(shape)
    N = np.empty(shape)
    CHI = np.empty(shape)
    SI = np.empty(shape)
    ARI = np.empty(shape)
    max_chi = (0, -1, -1)  # (chi, eps, m_s)
    max_si = (-1, -1, -1)  # (si, eps, m_s)
    for i in xrange(shape[0]):
        print str(M[i][0]) + "/" + str(M[-1][0])
        for j in xrange(shape[1]):
            k, noise, chi, si, ari = dbscan_params(E[i][j], M[i][j], data, true_labels)
            K[i][j] = k
            N[i][j] = noise
            CHI[i][j] = chi
            SI[i][j] = si
            ARI[i][j] = ari
            if sample_size * noise_limit >= noise:
                if chi > max_chi[0]:
                    max_chi = (chi, i, j)
                if si > max_si[0]:
                    max_si = (si, i, j)
    image_plot(E, M, K, "Clusters", "K", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, N, "Noise", "Noise", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, CHI, "Calinski-Harabasz index", "CHI", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, SI, "Silhouette Coefficient", "SI", max_chi, max_si, cmap=cmap, show_max=show_max)
    image_plot(E, M, ARI, "Adjusted Rand index", "ARI", max_chi, max_si, cmap=cmap, show_max=show_max)

    print "--"
    print_max("max_chi", max_chi, E, M, K, N, CHI, SI, ARI)
    print_max("max_si", max_si, E, M, K, N, CHI, SI, ARI)

    best_chi_labels = DBSCAN(eps=E[max_chi[1]][max_chi[2]], min_samples=M[max_chi[1]][max_chi[2]]).fit(data).labels_
    best_si_labels = DBSCAN(eps=E[max_si[1]][max_si[2]], min_samples=M[max_si[1]][max_si[2]]).fit(data).labels_

    if np.shape(data)[1] == 2:
        visualize(data, best_chi_labels, "Max CHI (" + "{:.1f}".format(CHI[max_chi[1]][max_chi[2]]) + ")")
        visualize(data, best_si_labels, "Max SI (" + "{:.3f}".format(SI[max_si[1]][max_si[2]]) + ")")
    else:
        print "Couldn't visualize clusterization result. Dimensions != 2"

    heat_map.run(data, best_chi_labels, title="Heatmap best CHI")
    heat_map.run(data, best_si_labels, title="Heatmap best SI")

    return best_chi_labels, best_si_labels
