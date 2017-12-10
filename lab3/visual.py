from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

MARKERS = {-1: ".", 0: ",", 1: "o", 2: "v", 3: "^", 4: "<",
           5: ">", 6: "*", 7: "x", 8: "+", 9: "4", 10: "_", 11: "s", 12: "p", 13: "P", 14: "1", 15: "h", 16: "H",
           17: "D", 18: "2", 19: "X", 20: "3", 21: "d", 22: "|", 23: "8"}


def cluster_coordinates(data, labels):
    x_clusters = defaultdict(list)
    y_clusters = defaultdict(list)
    for i in xrange(len(data)):
        x_clusters[labels[i]].append(data[i][0])
        y_clusters[labels[i]].append(data[i][1])
    return x_clusters, y_clusters


def visualize(data, labels, title="Visualization"):
    plt.figure()
    x_clusters, y_clusters = cluster_coordinates(data, labels)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i in xrange(len(x_clusters)):
        label = x_clusters.keys()[i]
        if (label >= 0):
            desc = "Cluster " + str(label)
        else:
            desc = "Noise"
        plt.scatter(x_clusters[label], y_clusters[label], marker=MARKERS[label], label=desc)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()

    # plt.show()
