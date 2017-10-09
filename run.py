import csv

from task1_3 import run1_3
from task2 import run2
from task4 import run4


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

run1_3(label_list, score_list)
# run2(label_list, score_list)
# run4(label_list, score_list)
