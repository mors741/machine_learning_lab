import csv

from lab2 import task1
from lab2.visual import visualize


def read_data():
    with open('data_v2-09.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        x1 = []
        x2 = []
        for row in reader:
            data.append((float(row[0]), float(row[1])))
            x1.append(float(row[0]))
            x2.append(float(row[1]))

        return data, x1, x2


data, x1, x2 = read_data()
# x1= [0.5, 0.5, 0.5]
# x2= [0, 0.5, 1]
visualize(x1, x2)
task1.run(x1, x2)
