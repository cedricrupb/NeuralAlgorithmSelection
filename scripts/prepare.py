import json
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def to_csv():
    with open("data.json", "r") as i:
        data = json.load(i)

    index = {}
    inv = []
    count = 0

    for D in data:
        for k in D.keys():
            if k not in index:
                index[k] = count
                inv.append(k)
                count += 1

    with open("feature_index.json", "w") as o:
        json.dump(index, o, indent=4)

    with open("data.csv", "w") as o:
        for D in data:
            for i in range(count):
                label = inv[i]
                c = 0
                if label in D:
                    c = D[label]
                o.write(str(c))
                if i == count - 1:
                    o.write("\n")
                else:
                    o.write("\t")


def train_test():
    my_data = np.genfromtxt("data.csv", delimiter="\t", dtype=np.int32)

    with open("labels.json", "r") as i:
        labels = json.load(i)

    data_train, data_test, label_train, label_test = train_test_split(
        my_data, labels, test_size=0.1, random_state=42
    )

    np.savetxt("train.csv", data_train, fmt="%d", delimiter="\t")
    np.savetxt("test.csv", data_test, fmt="%d", delimiter="\t")

    with open("train_labels.json", "w") as o:
        json.dump(label_train, o, indent=4)

    with open("test_labels.json", "w") as o:
        json.dump(label_test, o, indent=4)


def explore(tool1, tool2):
    my_data = np.genfromtxt("data.csv", delimiter="\t")

    with open("labels.json", "r") as i:
        labels = json.load(i)

    y_label = []

    for L in labels:
        l1 = 900
        l2 = 900

        for i, label in enumerate(L):
            if isinstance(label, list):
                if tool1 in label:
                    l1 = i
                if tool2 in label:
                    l2 = i
            elif label == tool1:
                l1 = i
            elif label == tool2:
                l2 = i
        if l1 < l2:
            y_label.append(2)
        elif l2 < l1:
            y_label.append(0)
        else:
            y_label.append(1)

    #y_label = np.array(y_label)

    color_mask = ["red", "grey", "green"]

    tsne = PCA(n_components=2)
    X = tsne.fit_transform(my_data)

    plt.scatter(X[:, 0], X[:, 1], c = [color_mask[_y] for _y in y_label])
    plt.show()

if __name__ == "__main__":
    explore("Klee", "ESBMC-incr")
