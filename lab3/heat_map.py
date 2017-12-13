import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances


def plot_heatmap(data, title='Heatmap'):
    fig, ax = plt.subplots()
    plt.imshow(data, interpolation='nearest', cmap=cm.viridis)
    cb = plt.colorbar()
    cb.set_label("Closeness")
    plt.title(title)
    plt.ylabel('Samples')
    plt.xlabel('Samples')
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    plt.show()


def run(data, distance='euclidean'):
    cluster_index = data.shape[1] - 1
    ind = np.argsort(data[:, cluster_index])
    data = data[ind]

    if distance == 'euclidean':
        dist = euclidean_distances(data, data)
    if distance == 'manhattan':
        dist = manhattan_distances(data, data)
    closeness = np.subtract(1, np.divide(dist, np.max(dist)))
    plot_heatmap(closeness)
