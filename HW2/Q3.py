import sys
from numpy import *
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
import matplotlib.patches as mpatches

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(
    where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_unscaled = data[train_idx[:6000], :].astype(float)
train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

validation_data_unscaled = data[train_idx[6000:], :].astype(float)
validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

test_data_unscaled = data[60000 + test_idx, :].astype(float)
test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


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
    etas = [x for x in range(1, 101)]
    T = 1000
    C = 1.0
    best_eta0 = 0
    best_accuracy = 0
    acs = []
    for eta0 in etas:
        accuracy = accuracyCalc(1.0 * eta0, C, T, validation_data, validation_labels)
        acs.append(accuracy)
        if accuracy > best_accuracy:
            best_eta0 = eta0
            best_accuracy = accuracy
    print('The best eta_0 is: ', best_eta0, 'with accuracy: ', best_accuracy)
    plt.figure(1)
    plt.plot(etas, acs)
    plt.xscale('log')
    plt.xlabel('$\eta_{0}$ value')
    plt.ylabel('Accuracy')
    plt.title('Different $\eta_{0}$ values vs. their accuracy')
    plt.figure(2)
    img_save = output + 'Q3_Section_A'
    plt.savefig(img_save)

    # Section B
    best_eta0 = 20
    C_list = [math.pow(math.sqrt(10), x) for x in range(-20, 22)]
    T = 1000
    best_C = 0
    best_accuracy = 0
    acs = []
    for C in C_list:
        accuracy = accuracyCalc(best_eta0, C, T, validation_data, validation_labels)
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

    # Section C
    best_eta0 = 20.0
    T = 20000
    best_C = 3.16 * math.pow(10, -5)
    w = ourSGDSVM(train_data, train_labels, best_C, best_eta0, T)
    plt.figure(3)
    plt.imshow(reshape(w, (28, 28)), interpolation='nearest', cmap='gray')
    plt.title('weight vector')
    img_save = output + 'Q3_Section_C_weight_vector'
    plt.savefig(img_save)

    # Section D
    best_accuracy = accuracyCalc(best_eta0, best_C, T, test_data, test_labels)
    print('The best accuracy on test set is: ', best_accuracy)


def accuracyCalc(eta0, C, T, set, labels):
    s = 0.0
    for i in range(10):
        w = ourSGDSVM(train_data, train_labels, C, eta0, T)
        s += testAccuracy(w, set, labels)
    return 1.0 * s / 10


def testAccuracy(w, set, labels):
    prediction = [np.dot(w, set[i]) for i in range(len(labels))]
    accuracy_for_validation = 1.0 * np.array(
        [0.0 if np.multiply(labels[i], prediction[i]) < 0 else 1.0 for i in
         range(len(labels))]).sum() / len(labels)
    return accuracy_for_validation


def ourSGDSVM(samples, labels, C, eta0, T):
    w = np.zeros(len(samples[0]), dtype='float64')
    for t in range(1, T + 1):
        i = np.random.randint(0, len(samples))
        prediction_result = False if np.multiply(labels[i], np.dot(w, samples[i])) < 1 else True
        eta_t = 1.0 * eta0 / t
        if not prediction_result:
            w = np.multiply((1 - eta_t), w) + np.multiply(eta_t * C * labels[i], samples[i])
    return w


if __name__ == '__main__':
    main(sys.argv[1:])
