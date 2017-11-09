import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator


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
    min_sens_minus_spec = (1, 0, 0, 0, 0)
    closest_to_perfect = (1, 0, 0, 0, 0)
    best_youdens = (0, 0, 0, 0, 0)
    best_f_score = (0, 0, 0, 0, 0)
    best_kappa = (0, 0, 0, 0, 0)

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
        ACC = (TP + TN)/float(P+N) # Accuracy

        roc_y.append(TPR)
        roc_x.append(FPR)

        if (TP + FP) != 0:
            precision = TP / float(TP + FP)
            recall = TPR

            pr_y.append(precision)
            pr_x.append(recall)

        sens_minus_spec = abs(TPR - 1 + FPR)
        if sens_minus_spec < min_sens_minus_spec[0]:
            min_sens_minus_spec = (sens_minus_spec, threshold, TPR, FPR, precision)

        perfect_dist = (1 - TPR)**2 + FPR**2
        if perfect_dist < closest_to_perfect[0]:
            closest_to_perfect = (perfect_dist, threshold, TPR, FPR, precision)

        j_index = TPR - FPR  # Youden's index (Youden's J statistic) sensitivity + specificity - 1
        if j_index > best_youdens[0]:
            best_youdens = (j_index, threshold, TPR, FPR, precision)

        if precision + recall != 0:
            f_score = (2 * precision * recall)/(precision + recall)
            if f_score > best_f_score[0]:
                best_f_score = (f_score, threshold, TPR, FPR, precision)

        ACC0 = ((TN+FN) * (TN+FP) + (TP+FP) * (TP+FN))/float(P+N)**2
        kappa = (ACC - ACC0)/(1 - ACC0)
        if kappa > best_kappa[0]:
            best_kappa = (kappa, threshold, TPR, FPR, precision)

    roc_x, roc_y = sort_coordinates(roc_x, roc_y)
    pr_x, pr_y = sort_coordinates(pr_x, pr_y, reverse_y=True)

    return roc_x, roc_y, pr_x, pr_y, [min_sens_minus_spec, closest_to_perfect, best_youdens, best_f_score, best_kappa]


def run1(label_list, score_list):
    roc_x, roc_y, pr_x, pr_y, best_thr = calculate_rot_pr_curves(label_list, score_list)

    roc_auc = np.trapz(roc_y, roc_x)
    pr_auc = np.trapz(pr_y, pr_x)

    plt.figure(1)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(roc_x, roc_y, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle='--', label="Random")

    plt.scatter(best_thr[0][3], best_thr[0][2], marker=9, color='g', label="|SENS-SPEC| min")
    plt.text(best_thr[0][3] + 0.04, best_thr[0][2] - 0.02, 'thr=' + "{:.5f}".format(best_thr[0][1]), color='g')

    plt.scatter(best_thr[1][3], best_thr[1][2], marker='x', color='y', label="Closest to perfect")
    plt.text(best_thr[1][3] + 0.02, best_thr[1][2] - 0.06, 'thr=' + "{:.5f}".format(best_thr[1][1]), color='y')

    plt.scatter(best_thr[2][3], best_thr[2][2], marker='o', color='r', label="Youden's max")
    plt.text(best_thr[2][3] + 0.02, best_thr[2][2] - 0.06, 'thr=' + "{:.5f}".format(best_thr[2][1]), color='r')

    plt.scatter(best_thr[3][3], best_thr[3][2], marker='d', color='m', label="F-score max")
    plt.text(best_thr[3][3] + 0.02, best_thr[3][2] - 0.04, 'thr=' + "{:.5f}".format(best_thr[3][1]), color='m')

    plt.scatter(best_thr[4][3], best_thr[4][2], marker=5, color='c', label="Kappa max")
    plt.text(best_thr[4][3] + 0.02, best_thr[4][2] - 0.09, 'thr=' + "{:.5f}".format(best_thr[4][1]), color='c')

    plt.text(.35, .15, 'AUC=' + "{:.5f}".format(roc_auc))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.legend()

    plt.figure(2)
    ax = plt.axes()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(pr_x, pr_y, label="PR curve")
    min_pr_y = min(pr_y)
    plt.plot([0, 1], [min_pr_y, min_pr_y], linestyle='--', label="Random")


    plt.scatter(best_thr[0][2], best_thr[0][4], marker=9, color='g', label="|SENS-SPEC| min")
    plt.text(best_thr[0][2] + 0.04, best_thr[0][4], 'thr=' + "{:.5f}".format(best_thr[0][1]), color='g')

    plt.scatter(best_thr[1][2], best_thr[1][4], marker='x', color='y', label="Closest to perfect")
    plt.text(best_thr[1][2] + 0.02, best_thr[1][4] + 0.02, 'thr=' + "{:.5f}".format(best_thr[1][1]), color='y')

    plt.scatter(best_thr[2][2], best_thr[2][4], marker='o', color='r', label="Youden's max")
    plt.text(best_thr[2][2] + 0.02, best_thr[2][4], 'thr=' + "{:.5f}".format(best_thr[2][1]), color='r')

    plt.scatter(best_thr[3][2], best_thr[3][4], marker='d', color='m', label="F-score max")
    plt.text(best_thr[3][2] + 0.02, best_thr[3][4], 'thr=' + "{:.5f}".format(best_thr[3][1]), color='m')

    plt.scatter(best_thr[4][2], best_thr[4][4], marker=5, color='c', label="Kappa max")
    plt.text(best_thr[4][2] + 0.02, best_thr[4][4] - 0.05, 'thr=' + "{:.5f}".format(best_thr[4][1]), color='c')

    plt.text(.5, .15, 'AUC=' + "{:.5f}".format(pr_auc))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve")
    plt.xlim([-0.05, 1.05])
    plt.ylim([0, 1.05])
    plt.legend()

    plt.show()
