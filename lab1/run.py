import csv

from lab1 import task1_3, task2, task4, task5
from lab1.visual import visualize


def read_data():
    with open('data_v1-09.csv', 'rb') as f:
        reader = csv.reader(f)
        reader.next()
        data = []
        x1 = []
        x2 = []
        label = []
        score = []
        for row in reader:
            data.append((float(row[0]), float(row[1]), int(row[2]), float(row[3])))
            x1.append(float(row[0]))
            x2.append(float(row[1]))
            label.append(int(row[2]))
            score.append(float(row[3]))

        return data, x1, x2, label, score


data, x1, x2, label_list, score_list = read_data()

visualize(data)
task1_3.run(label_list, score_list)
task2.run(label_list, score_list)
task4.run(label_list, score_list)
task5.run(label_list, score_list)
