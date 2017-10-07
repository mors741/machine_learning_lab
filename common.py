import csv
from random import shuffle


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


def get_samples(label_list, score_list, positive_size, negative_size):
    tuples = zip(score_list, label_list)
    shuffle(tuples)
    positive_count = 0
    negative_count = 0
    res = []
    for tpl in tuples:
        if tpl[1] > 0 and positive_count < positive_size:
            res.append(tpl)
            positive_count += 1
        elif tpl[1] < 0 and negative_count < negative_size:
            res.append(tpl)
            negative_count += 1

        if positive_count == positive_size and negative_count == negative_size:
            break
    return res
