import csv
import matplotlib.pyplot as plt
import numpy as np

from lab3 import dbscan
from lab3.visual import visualize


def read_data():
    with open('clust_data_9.txt', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        labels = []
        for row in reader:
            data.append((float(row[1]), float(row[2])))
            labels.append(int(row[3]))
        return data, labels


data, labels = read_data()
# visualize(data, labels)
dbscan.run(data)

plt.show()