import matplotlib.pyplot as plt
import numpy as np

from task1_3 import calculate_rot_pr_curves
from task2 import get_samples


def calculate_roc_pr_auc_dependency_from_size(label_list, score_list):
    sample_size_list = []
    roc_auc_list = []
    pr_auc_list = []

    for i in xrange(1, 51):
        sample_score_list, sample_label_list = get_samples(label_list, score_list, i, 6 * i)
        roc_x, roc_y, pr_x, pr_y, best_youdens = calculate_rot_pr_curves(sample_label_list, sample_score_list)
        sample_size_list.append(i * 7)
        roc_auc = np.trapz(roc_y, roc_x)
        pr_auc = np.trapz(pr_y, pr_x)
        roc_auc_list.append(roc_auc)
        pr_auc_list.append(pr_auc)

    return sample_size_list, roc_auc_list, pr_auc_list


def run4(label_list, score_list):
    sample_size_list, roc_auc_list, pr_auc_list = calculate_roc_pr_auc_dependency_from_size(label_list, score_list)

    plt.figure(1)
    plt.plot(sample_size_list, roc_auc_list)
    plt.xlabel("Sample size")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC dependency from Sample size")

    plt.figure(2)
    plt.plot(sample_size_list, pr_auc_list)
    plt.xlabel("Sample size")
    plt.ylabel("PR AUC")
    plt.title("PR AUC dependency from Sample size")

    plt.show()
