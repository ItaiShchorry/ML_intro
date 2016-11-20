import sys
from sklearn.datasets import fetch_mldata
import numpy.random
import math
import numpy as np
import operator
import matplotlib.pyplot as plt
from numpy import linalg as LA

def main():
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']
    idx = numpy.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :]
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :]
    test_labels = labels[idx[10000:]]
    print("banko")
    # section B
    k = 10
    precision = runKnnAndReturnPrecision(train[:1000], train_labels[:1000], test, test_labels, k)
    print(precision)

    pairs = np.array([np.array([(LA.norm(np.array(test[j]) - np.array(train[i])), train_labels[i])\
                      for i in range(5000)]) for j in range(test.shape[0])])

    # section C
    precisions = np.array([])
    ks_array = np.array([k for k in range(1, 101)])
    for k in ks_array:
        precision = secCandD(pairs, test_labels, k, 1000)
        precisions = np.append(precisions, precision)
    plt.axis([1, 100, 0, 1])
    plt.plot(ks_array, precisions, 'ro')
    plt.show()

    # section D
    training_samples = [x for x in range(100, 5100, 100)]
    k = 10
    precisions = np.array([])
    for n in training_samples:
        precision = secCandD(pairs, test_labels, k, n)
        precisions = np.append(precisions, precision)
    plt.axis([100, 5000, 0, 1])
    plt.plot(training_samples, precisions, 'ro')
    plt.show()

def runKnnAndReturnPrecision(train, train_labels, test , test_labels, k):
    cnt = 0
    for i in range(test.shape[0]):
        result = knn_estimate(train, train_labels, test[i], k)
        if result == test_labels[i]:
            cnt += 1
    return float(cnt) / test.shape[0]

def secCandD(pairs, test_labels, k, num_of_training):
    cnt = 0
    temp = pairs[:, 0:num_of_training]
    for i in range(test_labels.shape[0]):
        result = knn_with_pairs_ready(k, temp[i])
        if result == test_labels[i]:
            cnt += 1
    return float(cnt) / test_labels.shape[0]


def knn_estimate(images, labels, query, k):
    dists_and_labels = np.array([(LA.norm(np.array(query)-np.array(images[i])), labels[i]) for i in range(len(labels))])
    dists_and_labels_ordered = dists_and_labels[np.argsort(dists_and_labels[:, 0])]
    neighbors = np.array([int(dists_and_labels_ordered[i][1]) for i in range(k)])
    counts = np.bincount(neighbors)
    return np.argmax(counts)

def knn_with_pairs_ready(k, pairs):
    dists_and_labels_ordered = pairs[np.argsort(pairs[:, 0])]
    neighbors = np.array([int(dists_and_labels_ordered[i][1]) for i in range(k)])
    counts = np.bincount(neighbors)
    return np.argmax(counts)

def distAndLabel(image, label, test):
    return (LA.norm(np.array(test)-np.array(image)), label)

if __name__ == '__main__':
    sys.exit(main())