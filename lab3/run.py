import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from lab3 import dbscan, heat_map
from lab3.visual import visualize


def read_data():
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


data, labels, full_data = read_data()
# visualize(data, labels)
# dbscan.run(data, np.arange(0.02, 0.25, 0.05), xrange(6, 250, 20), cmap=cm.jet, show_max=True)
# dbscan.run(data, np.arange(0.02, 0.25, 0.003), xrange(6, 250, 3), cmap=cm.jet, show_max=True)
# dbscan.run(data, np.arange(0.075, 0.125, 0.001), xrange(50, 100, 1), cmap=cm.jet, show_max=True)


heat_map.run(np.array(full_data))

plt.show()