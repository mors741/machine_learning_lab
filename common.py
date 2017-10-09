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


def get_zipped_samples(label_list, score_list, positive_size, negative_size):
    score_list, label_list = get_samples(label_list, score_list, positive_size, negative_size)
    return zip(score_list, label_list)


def get_samples(label_list, score_list, positive_size, negative_size):
    tuples = zip(score_list, label_list)
    shuffle(tuples)
    positive_count = 0
    negative_count = 0
    res_score_list = []
    res_label_list = []
    for tpl in tuples:
        if tpl[1] > 0 and positive_count < positive_size:
            res_score_list.append(tpl[0])
            res_label_list.append(tpl[1])
            positive_count += 1
        elif tpl[1] < 0 and negative_count < negative_size:
            res_score_list.append(tpl[0])
            res_label_list.append(tpl[1])
            negative_count += 1

        if positive_count == positive_size and negative_count == negative_size:
            break
    return res_score_list, res_label_list


def sort_coordinates(x_list, y_list, reverse_y=False):
    def compare_coord_reverse_y(a, b):
        x_comp = cmp(a[0], b[0])
        if x_comp != 0:
            return x_comp
        else:
            return -cmp(a[1], b[1])

    new_x_list = []
    new_y_list = []
    sorted_pairs = sorted(((i, j) for i, j in zip(x_list, y_list)), cmp=compare_coord_reverse_y if reverse_y else cmp)
    for x, y in sorted_pairs:
        new_x_list.append(x)
        new_y_list.append(y)
    return new_x_list, new_y_list


def get_threshold_list(score_list):
    res = [min(score_list) - 0.0001]
    for sc in score_list:
        res.append(sc + 0.0001)
    return res


def get_positive_negative_count(label_list):
    P = len(filter(lambda x: x > 0, label_list))
    N = len(label_list) - P
    return P, N


def calculate_rot_pr_curves(label_list, score_list):
    threshold_list = get_threshold_list(score_list)

    P, N = get_positive_negative_count(label_list)

    roc_x = []
    roc_y = []
    pr_x = []
    pr_y = []
    best_youdens = (0, 0, 0, 0, 0)

    for threshold in threshold_list:
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(0, len(score_list)):
            if score_list[i] > threshold:
                if label_list[i] > 0:
                    TP += 1
                else:
                    FP += 1
        FN = P - TP
        TN = N - FP

        TPR = TP / float(P)  # True Positive Rate
        FPR = FP / float(N)  # False Positive Rate

        roc_y.append(TPR)
        roc_x.append(FPR)

        if (TP + FP) != 0:
            precision = TP / float(TP + FP)
            recall = TPR

            pr_y.append(precision)
            pr_x.append(recall)

        j_index = TPR - FPR  # Youden's index (Youden's J statistic) sensitivity + specificity - 1
        if (j_index > best_youdens[0]):
            best_youdens = (j_index, threshold, TPR, FPR, precision)

    roc_x, roc_y = sort_coordinates(roc_x, roc_y)
    pr_x, pr_y = sort_coordinates(pr_x, pr_y, reverse_y=True)

    return roc_x, roc_y, pr_x, pr_y, best_youdens
