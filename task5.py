import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


from task4 import calculate_roc_pr_auc_dependency_from_size


def corcoef(a_list, b_list):
    a_avg = np.average(a_list)
    b_avg = np.average(b_list)
    covar = 0
    sq_disp_a = 0
    sq_disp_b = 0
    for i in xrange(0, len(a_list)):
        a_dif = (a_list[i] - a_avg)
        b_dif = (b_list[i] - b_avg)
        covar += a_dif * b_dif
        sq_disp_a += a_dif ** 2
        sq_disp_b += b_dif ** 2

    return covar / np.sqrt(sq_disp_a * sq_disp_b)


def run5(label_list, score_list):
    sample_size_list, roc_auc_list, pr_auc_list = calculate_roc_pr_auc_dependency_from_size(label_list, score_list)

    correlation = corcoef(roc_auc_list, pr_auc_list)
    print "ROC AUC & PR AUC =", correlation
    print "SIZE & ROC AUC =", corcoef(sample_size_list, roc_auc_list)
    print "SIZE & PR AUC =", corcoef(sample_size_list, pr_auc_list)

    print "--- scipy result ---"
    print "ROC AUC & PR AUC =", stats.pearsonr(roc_auc_list, pr_auc_list)
    print "SIZE & ROC AUC =", stats.pearsonr(sample_size_list, roc_auc_list)
    print "SIZE & PR AUC =", stats.pearsonr(sample_size_list, pr_auc_list)

    plt.scatter(roc_auc_list, pr_auc_list)
    plt.xlabel("ROC AUC")
    plt.ylabel("PR AUC")
    ax = plt.axes()
    plt.text(.65, .2, 'Correlation=' + "{:.5f}".format(correlation), transform=ax.transAxes)
    plt.title("ROC AUC and PR AUC Scatter plot")

    plt.show()
