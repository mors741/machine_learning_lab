import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score, silhouette_score

from lab3 import heat_map
from lab3.visual import visualize


def k_means(data, n_clusters):
    cluster_result = KMeans(n_clusters=n_clusters, init='k-means++').fit(data)
    labels = cluster_result.labels_
    centers = cluster_result.cluster_centers_
    interia = cluster_result.inertia_
    return labels, centers, interia


def draw_plot(x, y, x_label, y_label):
    plt.figure()
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.title(y_label + '(' + x_label + ')')
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def run(data, max_clusters, vis_range):
    interias = []
    calinski_indexes = []
    silhouette_indexes = []

    for n_cluster in range(2, max_clusters + 1):
        labels, centers, interia = k_means(data, n_cluster)
        interias.append(interia)
        calinski_indexes.append(calinski_harabaz_score(data, labels))
        silhouette_indexes.append(silhouette_score(data, labels, metric='euclidean'))

        if np.shape(data)[1] == 2 and n_cluster in vis_range:
            visualize(data, labels)
            heat_map.run(data, labels, title="K="+str(n_cluster))

    draw_plot(range(2, max_clusters + 1), interias, "K", "S")
    draw_plot(range(2, max_clusters + 1), calinski_indexes, "K", "CHI")
    draw_plot(range(2, max_clusters + 1), silhouette_indexes, "K", "SI")
