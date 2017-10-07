from scipy import stats

from common import read_data, get_samples

SAMPLE_SIZE = 40
EPSILON = 0.000001
U_CRITICAL = 557  # 1%


def ranked(samples):
    res = []
    tmp_res = []
    cur_rank = 1
    sorted_samples = sorted(samples, key=lambda x: x[0])
    for tpl in sorted_samples:
        if len(tmp_res) > 0 and abs(tmp_res[0][0] - tpl[0]) > EPSILON:
            tmp_rank = (cur_rank * 2 - len(tmp_res) - 1) / float(2)
            for tmp_tpl in tmp_res:
                res.append((tmp_tpl[0], tmp_tpl[1], tmp_rank))
            tmp_res = []

        tmp_res.append((tpl[0], tpl[1], cur_rank))
        cur_rank += 1
    tmp_rank = (cur_rank * 2 - len(tmp_res) - 1) / float(2)
    for tmp_tpl in tmp_res:
        res.append((tmp_tpl[0], tmp_tpl[1], tmp_rank))
    return res


def sum_ranks(ranked_samples):
    positive_sum = 0
    negative_sum = 0
    for tpl in ranked_samples:
        if tpl[1] > 0:
            positive_sum += tpl[2]
        else:
            negative_sum += tpl[2]
    return positive_sum, negative_sum


def u_emp_value(positive_sum, negative_sum):
    return SAMPLE_SIZE ** 2 + (SAMPLE_SIZE * (SAMPLE_SIZE + 1)) / 2 - max(positive_sum, negative_sum)


def to_lists(samples):
    x = []
    y = []
    for tpl in samples:
        if tpl[1] > 0:
            x.append(tpl[0])
        else:
            y.append(tpl[0])
    return x, y


def task2():
    data, x1, x2, label_list, score_list = read_data()
    samples = get_samples(label_list, score_list, SAMPLE_SIZE, SAMPLE_SIZE)
    ranked_samples = ranked(samples)
    positive_sum, negative_sum = sum_ranks(ranked_samples)
    u_emp = u_emp_value(positive_sum, negative_sum)

    print "samples =", samples
    print "ranked_samples =", ranked_samples
    print "positive_sum =", positive_sum, negative_sum
    print "u_emp =", u_emp

    if u_emp < U_CRITICAL:
        print "hypothesis accepted"
    else:
        print "hypothesis rejected"

    x, y = to_lists(samples)

    statistic, pvalue = stats.mannwhitneyu(x, y)
    print "--- scipy result ---"
    print "u_emp = ", statistic
    print "pvalue = ", pvalue


task2()
