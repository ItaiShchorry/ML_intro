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

    etas = [1e-20,1e-19,1e-18,1e-17,1e-16,1e-15\
        ,1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e-0]
    T = 5000
    C = 1.0
    best_eta = 0
    best_accuracy = 0
    acs = []
    for eta in etas:
        accuracy = accuracyCalc(1.0 * eta, C, T, validation_data, validation_labels, quadratic)
        acs.append(accuracy)
        if accuracy > best_accuracy:
            best_eta = eta
            best_accuracy = accuracy
        print('done for eta=',eta, 'accuracy=', accuracy)
    print('The best eta is: ', best_eta, 'with accuracy: ', best_accuracy)
    plt.figure(1)
    plt.plot(etas, acs)
    plt.xlabel('$\eta$ value')
    plt.xscale('log')
    plt.ylabel('Accuracy')
    plt.title('Different $\eta$ values vs. their accuracy')
    img_save = output + 'Q3_Section_A_eta'
    plt.savefig(img_save)

    C_list = [1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10]
    best_C = 0
    best_accuracy = 0
    acs = []
    for C in C_list:
        accuracy = accuracyCalc(best_eta, C, T, validation_data, validation_labels, quadratic)
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
    img_save = output + 'Q3_Section_A_C'
    plt.savefig(img_save)

    # Section C
    best_accuracy = accuracyCalc(best_eta, best_C, T, test_data, test_labels, quadratic)
    print('The best accuracy on test set is: ', best_accuracy)


def accuracyCalc(eta, C, T, set, labels, kernel):
    s = 0.0
    for i in range(10):
        alphas = ourKernelSGDSVM(train_data, train_labels, C, eta, T, kernel)
        s += testAccuracy(alphas, set, labels, kernel)
    return 1.0 * s / 10


def testAccuracy(alphas, set, labels, kernel):
    accuracy_for_validation = 0
    for i in range(set.shape[0]):
        prediction = predict(alphas, kernel, set[i])
        if prediction == labels[i]:
            accuracy_for_validation += 1.0

    return accuracy_for_validation/len(labels)


def ourKernelSGDSVM(samples, labels, eta, C, T, kernel):
    alphas = np.zeros((samples.shape[0], 10), dtype='float64')
    for t in range(1, T + 1):
        i = np.random.randint(0, len(samples))
        xi = samples[i]
        yi = int(labels[i])
        indicator_vec = [int(j != yi) for j in K]
        k = np.array([kernel(samples[n], xi) for n in range(samples.shape[0])])
        penalty_vec = [np.dot(k,np.subtract(alphas[:,j], alphas[:,yi])) + indicator_vec[j] for j in K]
        max_j = np.argmax(penalty_vec)
        alphas[:,:] = np.multiply(alphas[:,:], (1 - eta))
        if max_j != yi:
            alphas[i, max_j] = np.subtract(alphas[i, max_j], eta*C)
            alphas[i, yi] = np.subtract(alphas[i, yi], -eta*C)
    return alphas

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def predict(alphas, kernel, sample):
    k = np.array([kernel(train_data[n], sample) for n in range(train_data.shape[0])])
    predictions = np.array([np.dot(alphas[:,j],k) for j in K])
    return np.argmax(predictions)

def lin(x,y):
    return np.dot(x,y)

def quadratic(x,y):
    return np.square(np.dot(x, y) + 1)

if __name__ == '__main__':
    main(sys.argv[1:])