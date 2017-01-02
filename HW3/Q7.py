import sys
import os
from numpy import *
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from numpy import linalg as LA
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
import matplotlib.patches as mpatches
matplotlib.use('Agg')
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

train_idx = numpy.random.RandomState(0).permutation(range(60000))

train_data_size = 50000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = labels[train_idx[:train_data_size]]

validation_data_unscaled = data[train_idx[train_data_size:60000], :].astype(float)
validation_labels = labels[train_idx[train_data_size:60000]]

test_data_unscaled = data[60000:, :].astype(float)
test_labels = labels[60000:]

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
K = [i for i in range(10)]

def main(args):
    # output path:
    if len(args) == 1:
        output = args[0] + '/'
        if not os.path.exists(output):
            print("Path does not exist!")
            sys.exit(2)
    elif len(args) > 1:
        print("usage: Q3.py <output_path>")
        sys.exit(2)
    else:
        output = ''
        # Section A

    etas = [x for x in range(1, 100)]
    T = 1000
    C = 1.0
    best_eta = 0
    best_accuracy = 0
    acs = []
    for eta in etas:
        accuracy = accuracyCalc(1.0 * eta, C, T, validation_data, validation_labels)
        acs.append(accuracy)
        if accuracy > best_accuracy:
            best_eta = eta
            best_accuracy = accuracy
    print('The best eta_0 is: ', best_eta, 'with accuracy: ', best_accuracy)
    plt.figure(1)
    plt.plot(etas, acs)
    plt.xlabel('$\eta_{0}$ value')
    plt.ylabel('Accuracy')
    plt.title('Different $\eta_{0}$ values vs. their accuracy')
    img_save = output + 'Q3_Section_A'
    plt.savefig(img_save)

    # Section B
    C_list = [math.pow(math.sqrt(10), x) for x in range(-20, 22)]
    T = 1000
    best_C = 0
    best_accuracy = 0
    acs = []
    for C in C_list:
        accuracy = accuracyCalc(best_eta, C, T, validation_data, validation_labels)
        acs.append(accuracy)
        if accuracy > best_accuracy:
            best_C = C
            best_accuracy = accuracy
    print('The best C is: ', best_C, 'with accuracy: ', best_accuracy)
    plt.figure(2)
    plt.plot(C_list, acs)
    plt.xscale('log')
    plt.xlabel('C value')
    plt.ylabel('Accuracy')
    plt.title('Different C values vs. their accuracy')
    img_save = output + 'Q3_Section_B'
    plt.savefig(img_save)


def accuracyCalc(eta, C, T, set, labels):
    s = 0.0
    for i in range(10):
        weights = ourNonKernelSGDSVM(train_data, train_labels, C, eta, T)
        s += testAccuracy(weights, set, labels)
    return 1.0 * s / 10


def testAccuracy(weights, set, labels):
    accuracy_for_validation = 0
    for i in range(set.shape[0]):
        indicator_vec = [1 if j != labels[i] else 0 for j in K]
        penalty_vec = [np.dot(set[i], weights[j]) \
                   - np.dot(set[i], weights[int(labels[i])]) + indicator_vec[j] for j in K]
        prediction = np.argmax(penalty_vec)
        if prediction == labels[i]:
            accuracy_for_validation += 1.0

    return accuracy_for_validation/len(labels)


def ourNonKernelSGDSVM(samples, labels, C, eta, T):
    weights = [np.zeros(len(samples[0]), dtype='float64')] * 10
    for t in range(1, T + 1):
        i = np.random.randint(0, len(labels))
        indicator_vec = [1 if j != labels[i] else 0 for j in K]
        penalty_vec = [np.dot(samples[i], weights[j])
                       - np.dot(samples[i], weights[int(labels[i])]) + indicator_vec[j] for j in K]
        max_j = np.argmax(penalty_vec)
        for j in K:
            weights[j] = (1 - eta) * weights[j]
        if max_j != labels[i]:
            weights[max_j] -= eta * C * samples[i]
            weights[int(labels[i])] += eta * C * samples[i]
    return weights

def ourKernelSGDSVM(samples, labels, eta, C, T):
    weights = [np.zeros(len(samples[0]), dtype='float64')] * 10
    for t in range(1, T + 1):
        i = np.random.randint(0, len(samples))
        indicator_vec = [1 if j != labels[i] else 0 for j in K]
        penalty_vec = [np.dot(samples[i], weights[j])\
                       - np.dot(samples[i], weights[labels[i]]) + indicator_vec[j] for j in K]
        max_j = np.max(penalty_vec)
        weights[max_j] -= eta * C * samples[i]
        weights[labels[i]] += eta * C * samples[i]
    return weights

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


if __name__ == '__main__':
    main(sys.argv[1:])