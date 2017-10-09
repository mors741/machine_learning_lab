import matplotlib.pyplot as plt
import numpy as np


def get_threshold_list(score_list):
    res = [min(score_list) - 0.0001]
    for sc in score_list:
        res.append(sc + 0.0001)
    return res


def get_positive_negative_count(label_list):
    P = len(filter(lambda x: x > 0, label_list))
    N = len(label_list) - P
    return P, N


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


def run1_3(label_list, score_list):
    roc_x, roc_y, pr_x, pr_y, best_youdens = calculate_rot_pr_curves(label_list, score_list)

    roc_auc = np.trapz(roc_y, roc_x)
    pr_auc = np.trapz(pr_y, pr_x)

    plt.figure(1)
    plt.plot(roc_x, roc_y, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Random")
    plt.scatter(best_youdens[3], best_youdens[2], marker='o', color='r', label="Youden's max")
    plt.text(best_youdens[3] + 0.02, best_youdens[2] - 0.06, 'thr=' + "{:.5f}".format(best_youdens[1]))
    plt.text(.5, .2, 'AUC=' + "{:.5f}".format(roc_auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend()

    plt.figure(2)
    plt.plot(pr_x, pr_y, label="PR curve")
    plt.scatter(best_youdens[2], best_youdens[4], marker='o', color='r', label="Youden's max")
    plt.text(best_youdens[2] - 0.02, best_youdens[4] + 0.04, 'thr=' + "{:.5f}".format(best_youdens[1]))
    plt.text(.5, .2, 'AUC=' + "{:.5f}".format(pr_auc))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.ylim([-0.05, 1.05])
    plt.legend()

    plt.show()
