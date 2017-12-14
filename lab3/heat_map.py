import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import manhattan_distances


def plot_heatmap(data, title):
    plt.figure()
    plt.imshow(data, interpolation='nearest', cmap=cm.viridis)
    cb = plt.colorbar()
    cb.set_label("Closeness")
    plt.title(title)
    plt.ylabel('Samples')
    plt.xlabel('Samples')


def label_changes(sorted_labels):
    res = []
    for i in xrange(len(sorted_labels)-1):
        if sorted_labels[i] != sorted_labels[i+1]:
            print res.append(i)
    return res


def run(data, labels, distance='euclidean', title='Heatmap'):
    data = np.array(data)
    labels = np.array(labels)
    noise_inds = np.where(labels == -1)
    data = np.delete(data, noise_inds, 0)
    labels = np.delete(labels, noise_inds)
    ind = np.argsort(labels)
    data = data[ind]

    if distance == 'euclidean':
        dist = euclidean_distances(data, data)
    if distance == 'manhattan':
        dist = manhattan_distances(data, data)
    closeness = np.subtract(1, np.divide(dist, np.max(dist)))
    plot_heatmap(closeness, title)
