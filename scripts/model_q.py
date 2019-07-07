import numpy as np
import argparse
import random

from sklearn.metrics.pairwise import cosine_similarity


def load_labels():
    clusters = []
    labels = []
    with open('cluster.txt', "r") as i:
        for s in i:
            label, cluster = s.split(':')
            try:
                cluster = int(cluster[:-1])
            except Exception:
                cluster = random.randint(5, 10000)
            labels.append(label)
            clusters.append(cluster)
    return labels, clusters


parser = argparse.ArgumentParser()
parser.add_argument('type')

args = parser.parse_args()

t = int(args.type)

labels, _ = load_labels()

X = np.loadtxt('model.csv')
sim = cosine_similarity(X)

print("Label: %s" % labels[t])
x = sim[t, :]
p = np.argsort(x)[::-1][1:11]

print("top-10:")
for i in range(p.shape[0]):
    print("\t %d: %s [%f]" % (i, labels[p[i]], x[p[i]]))
