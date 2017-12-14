import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.cluster.supervised import adjusted_rand_score

from lab3 import dbscan, heat_map, k_means
from lab3.visual import visualize


def read_test_data():
    with open('clust_data_9.txt', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        labels = []
        for row in reader:
            data.append((float(row[1]), float(row[2])))
            labels.append(int(row[3]))
        return data, labels


def read_real_data():
    with open('real_data.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        labels = []
        for row in reader:
            data.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))
            labels.append(int(row[5]))
        return data, labels


def compare_ari(data, k, eps, m_s):
    k_means_labels = KMeans(n_clusters=k, init='k-means++').fit(data).labels_
    dbscan_labels = DBSCAN(eps=eps, min_samples=m_s).fit(data).labels_

    k_means_labels = np.array(k_means_labels)
    dbscan_labels = np.array(dbscan_labels)

    heat_map.run(data, k_means_labels)
    heat_map.run(data, dbscan_labels)

    noise_inds = np.where(dbscan_labels == -1)
    k_means_labels = np.delete(k_means_labels, noise_inds, 0)
    dbscan_labels = np.delete(dbscan_labels, noise_inds, 0)

    # heat_map.run(data, k_means_labels)
    # heat_map.run(data, dbscan_labels)

    return adjusted_rand_score(k_means_labels, dbscan_labels)


def investigate_test_data():
    data, labels = read_test_data()
    visualize(data, labels)
    heat_map.run(data, labels)

    k_means.run(data, labels, 25, [2, 10, 18])

    # dbscan.run(data, np.arange(0.02, 0.05, 0.001), xrange(2, 30, 1), labels, cmap=cm.jet, show_max=True, noise_limit=0.03)


    dbscan.run(data, np.arange(0.02, 0.25, 0.01), xrange(6, 250, 8), labels, cmap=cm.jet, show_max=True, noise_limit=0.03)
    # dbscan.run(data, np.arange(0.03, 0.08, 0.01), xrange(2, 10, 1), labels, cmap=cm.jet, show_max=True, noise_limit=0.03)
    # dbscan.run(data, np.arange(0.02, 0.25, 0.003), xrange(6, 250, 3), labels, cmap=cm.jet, show_max=True)


    # dbscan.run(data, np.arange(0.05, 0.061, 0.001), xrange(22, 28, 1), labels, cmap=cm.jet, show_max=True, noise_limit=0.03)

    # dbscan.visualize_dbscan(0.05, 7, data, "DBSCAN [eps=0.05, m_s=7]")
    # dbscan.visualize_dbscan(0.06, 26, data, "DBSCAN [eps=0.06, m_s=26]")
    # dbscan.visualize_dbscan(0.191, 156, data, "DBSCAN [eps=0.191, m_s=156]")
    # dbscan.visualize_dbscan(0.06, 27, data, "DBSCAN [eps=0.06, m_s=27]")

    print compare_ari(data, 2, 0.191, 156)
    # print compare_ari(data, 10, 0.06, 27)
    # print compare_ari(data, 18, 0.05, 7)



    plt.show()


def investigate_real_data():
    data, labels = read_real_data()
    heat_map.run(data, labels)
    k_means.run(data, labels, 25, [2, 4, 11, 13])
    dbscan.run(data, np.arange(0.02, 0.4, 0.01), xrange(2, 20, 1), labels, cmap=cm.jet, show_max=True, noise_limit=0.03)
    # dbscan.visualize_dbscan(0.19, 6, data, "DBSCAN [eps=0.19, m_s=6]")
    # dbscan.visualize_dbscan(0.31, 6, data, "DBSCAN [eps=0.19, m_s=6]")

    print compare_ari(data, 2, 0.34, 2)
    # print compare_ari(data, 4, 0.34, 2)
    # print compare_ari(data, 11, 0.34, 2)


    plt.show()


investigate_test_data()
investigate_real_data()
