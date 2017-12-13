import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from lab3 import dbscan, heat_map


def read_test_data():
    with open('clust_data_9.txt', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        full_data = []
        labels = []
        for row in reader:
            full_data.append([float(row[1]), float(row[2]), int(row[3])])
            data.append((float(row[1]), float(row[2])))
            labels.append(int(row[3]))
        return data, labels, full_data


def read_real_data():
    with open('real_data.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        full_data = []
        labels = []
        for row in reader:
            full_data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), int(row[5])])
            data.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))
            labels.append(int(row[5]))
        return data, labels, full_data


def investigate_test_data():
    data, labels, full_data = read_test_data()
    # visualize(data, labels)
    dbscan.run(data, np.arange(0.02, 0.25, 0.05), xrange(6, 250, 20), cmap=cm.jet, show_max=True)
    # dbscan.run(data, np.arange(0.02, 0.25, 0.003), xrange(6, 250, 3), cmap=cm.jet, show_max=True)

    heat_map.run(np.array(full_data))

    plt.show()


def investigate_real_data():
    data, labels, full_data = read_real_data()
    dbscan.run(data, np.arange(0.02, 0.4, 0.01), xrange(2, 20, 1), cmap=cm.jet, show_max=True)

    heat_map.run(np.array(full_data))

    plt.show()


# investigate_test_data()
investigate_real_data()
