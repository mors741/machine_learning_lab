from sklearn.cluster import DBSCAN

from lab3.visual import visualize


def run(data):
    db = DBSCAN(eps=0.04, min_samples=10).fit(data)
    visualize(data, db.labels_)
