from sklearn.metrics.cluster import adjusted_rand_score


def get_ARI_score(data1, data2, label_index1, label_index2):
    labels1 = data1[:, label_index1]
    labels2 = data2[:, label_index2]
    ari_score = calculate_ARI(labels1, labels2)
    return ari_score


def calculate_ARI(labels_true, labels_predicted):
    return adjusted_rand_score(labels_true, labels_predicted)
