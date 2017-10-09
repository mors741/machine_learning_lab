import matplotlib.pyplot as plt
import numpy as np

from common import read_data, calculate_rot_pr_curves


def task1_3():
    data, x1, x2, label_list, score_list = read_data()

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


task1_3()
