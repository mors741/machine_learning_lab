import csv

import task2_1
from task1_1 import run1
from task1_2 import run2
from task1_4 import run4
from task1_5 import run5
from visual1 import visualize1
from visual2 import visualize2


def read_data1():
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

def read_data2():
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


# data, x1, x2, label_list, score_list = read_data1()
#
# visualize1(data)
# run1(label_list, score_list)
# run2(label_list, score_list)
# run4(label_list, score_list)
# run5(label_list, score_list)

data, x1, x2 = read_data2()
# x1= [0.5, 0.5, 0.5]
# x2= [0, 0.5, 1]
# visualize2(x1, x2)
task2_1.run1(x1, x2)
