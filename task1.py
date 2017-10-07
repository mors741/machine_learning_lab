import matplotlib.pyplot as plt

import numpy as np

from common import read_data


def get_threshold_list(score_list):
    res = [min(score_list) - 0.0001]
    for sc in score_list:
        res.append(sc + 0.0001)
    return res


def get_positive_negative_count(label_list):
    P = len(filter(lambda x: x > 0, label_list))
    N = len(label_list) - P
    return P, N


def compare_coord_reverse_y(a, b):
    x_comp = cmp(a[0], b[0])
    if x_comp != 0:
        return x_comp
    else:
        return -cmp(a[1], b[1])


def sort_coordinates(x_list, y_list, reverse_y=False):
    new_x_list = []
    new_y_list = []
    sorted_pairs = sorted(((i, j) for i, j in zip(x_list, y_list)), cmp=compare_coord_reverse_y if reverse_y else cmp)
    for x, y in sorted_pairs:
        new_x_list.append(x)
        new_y_list.append(y)
    return new_x_list, new_y_list


def task1():
    data, x1, x2, label_list, score_list = read_data()

    threshold_list = get_threshold_list(score_list)

    P, N = get_positive_negative_count(label_list)

    roc_x = []
    roc_y = []
    pr_x = []
    pr_y = []

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

    roc_x, roc_y = sort_coordinates(roc_x, roc_y);
    pr_x, pr_y = sort_coordinates(pr_x, pr_y, reverse_y=True);
    roc_auc = np.trapz(roc_y, roc_x)
    pr_auc = np.trapz(pr_y, pr_x)

    plt.figure(1)
    plt.plot(roc_x, roc_y, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle='--', label="random")
    plt.text(.5, .2, 'AUC=' + "{:.5f}".format(roc_auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()

    plt.figure(2)
    plt.plot(pr_x, pr_y, label="PR curve")
    plt.text(.5, .2, 'AUC=' + "{:.5f}".format(pr_auc))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.ylim([-0.05, 1.05])
    plt.legend()

    plt.show()


task1()
