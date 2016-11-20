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
    # section B
    k = 10
    precision = runKnnAndReturnPrecision(train[:1000], train_labels[:1000], test, test_labels, k)
    print('Precision for section B:', precision)

    # Creating all pairs of distances of the training pictures with the test pictures
    # It will save running time for sections C and D.
    pairs = np.array([np.array([(LA.norm(np.array(test[j]) - np.array(train[i])), train_labels[i])\
                      for i in range(5000)]) for j in range(test.shape[0])])

    # Plotting configurations for sections C and D.
    ax0 = plt.subplot(211)
    ax0.set_xlabel('k')
    ax0.set_ylabel('precision')
    plt.text(0.5, 0.5, 'Section C',
         horizontalalignment='center',
         verticalalignment='center',
         transform=ax0.transAxes)
    ax1 = plt.subplot(212)
    ax1.set_ylabel('precision')
    ax1.set_xlabel('training samples')
    plt.text(0.5, 0.5, 'Section D',
             horizontalalignment='center',
             verticalalignment='center',
             transform=ax1.transAxes)

    # section C
    precisions = np.array([])
    ks_array = np.array([k for k in range(1, 101)])
    max = 0
    max_k = 0
    for k in ks_array:
        precision = secCandD(pairs, test_labels, k, 1000)
        precisions = np.append(precisions, precision)
        if precision > max:
            max_k = k
            max = precision
    print('K = {} gives the maximum precision in section C.'.format(max_k))
    ax0.plot(ks_array, precisions)

    # section D
    training_samples = [x for x in range(100, 5100, 100)]
    k = 10
    precisions = np.array([])
    for n in training_samples:
        precision = secCandD(pairs, test_labels, k, n)
        precisions = np.append(precisions, precision)
    ax1.plot(training_samples, precisions)
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