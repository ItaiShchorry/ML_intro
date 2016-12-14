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
    where((labels[:60000] == neg) | (labels[:60000] == pos))[0])  # TODO why is the [0] at the end?
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
        print("usage: hw2.py <output_path>")
        sys.exit(2)
    else:
        output = ''
    C_list = [math.pow(10,x) for x in range(-10,11)]
    best_accuracy = 0
    best_C = 0
    for c in C_list:
        clf = svm.LinearSVC(C=c, loss='hinge', fit_intercept=False)
        clf.fit(train_data, train_labels)
        prediction = clf.predict(validation_data)
        accuracy = 1.0*np.sum(np.array([1.0 if prediction[i] == validation_labels[i] else 0.0 for i in range(validation_labels.shape[0])]))/validation_labels.shape[0]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = c
    print('We can see that the best C value for validation set in the given range is ', best_C)
    print('Trying now the range: from 10^-8 to 10^-6 steps of 5*10^-8')

    C_list = [x for x in np.arange(math.pow(10,-8),math.pow(10,-6)+math.pow(10,-8),math.pow(10,-8))]
    best_accuracy = 0
    best_C = 0
    validation_set_accuracies = []
    training_set_accuracies = []
    for c in C_list:
        clf = svm.LinearSVC(C=c, loss='hinge', fit_intercept=False)
        clf.fit(train_data, train_labels)
        prediction_for_validation = clf.predict(validation_data)
        accuracy_for_validation = 1.0*np.sum(np.array([1.0 if prediction_for_validation[i] == validation_labels[i] else 0.0 for i in range(validation_labels.shape[0])]))/validation_labels.shape[0]
        prediction_for_training = clf.predict(train_data)
        accuracy_for_training = 1.0*np.sum(np.array([1.0 if prediction_for_training[i] == train_labels[i] else 0.0 for i in range(train_labels.shape[0])]))/train_labels.shape[0]
        validation_set_accuracies.append(accuracy_for_validation)
        training_set_accuracies.append(accuracy_for_training)
        if accuracy_for_validation > best_accuracy:
            best_accuracy = accuracy_for_validation
            best_C = c

    print('We can see that the best C value for validation set in the given range is ', best_C)

    print('best C value is ', best_C)
    plt.figure(1)
    plt.xlabel('C value')
    plt.ylabel('Accuracy')
    plt.title('Different C values vs. their accuracy')
    plt.xscale('log')
    green_patch = mpatches.Patch(color='green', label='Accuracy for validation set')
    red_patch = mpatches.Patch(color='red', label='Accuracy for training set')
    plt.plot(C_list, validation_set_accuracies, 'g', C_list, training_set_accuracies, 'r')
    plt.legend(handles=[green_patch, red_patch])
    img_save = output + 'Q2_Section_A'
    plt.savefig(img_save)
    clf = svm.LinearSVC(C=best_C, loss='hinge', fit_intercept=False)
    clf.fit(train_data, train_labels)
    plt.figure(2)
    plt.imshow(reshape(clf.coef_, (28, 28)), interpolation='nearest', cmap='gray')
    plt.title('weight vector')
    img_save = output + 'Q2_weight_vector'
    plt.savefig(img_save)
    prediction_for_test = clf.predict(test_data)
    accuracy_for_training = 1.0 * np.sum(np.array([1.0 if prediction_for_test[i] == test_labels[i] else 0.0 for i in range(test_labels.shape[0])]))/test_labels.shape[0]
    print('The accuracy of the linear SVM with the best C on the test set is: ', accuracy_for_training)

    etas = [x for x in np.arange(0.0001, 0.001, 0.0002)]
    T = 1000
    C = 1
    best_eta0 = 0
    best_accuracy = 0
    for eta0 in etas:
        accuracy = accuracyCalc(eta0, C, T)
        if accuracy > best_accuracy:
            best_eta0 = eta0
            best_accuracy = accuracy
    print('The best eta0 is: ', best_eta0, 'with accuracy: ', best_accuracy)

def accuracyCalc(eta0, C, T):
    s = 0
    for i in range(10):
        w = ourSGDSVM(train_data, train_labels, C, eta0, T)
        s = testAccuracy(w)
    return 1.0*s/10

def testAccuracy(w):
    validation_normalized_labels = [-1.0 if validation_labels[i] == neg else 1.0 for i in range(len(validation_labels))]
    prediction_for_validation = [np.dot(w, validation_data[i]) for i in range(len(validation_labels))]
    accuracy_for_validation = 1.0 * np.sum(np.array(
        [1.0 if prediction_for_validation[i]*validation_normalized_labels[i]>0 else 0.0 for i in
         range(len(validation_normalized_labels))]))/len(validation_normalized_labels)
    return accuracy_for_validation


def ourSGDSVM(samples, labels, C, eta0, T):
    w = np.zeros(len(samples[0]), dtype=float)
    for t in range(1, T+1):
        i = np.random.randint(0, len(samples))
        eta_t = eta0/t
        if np.multiply(labels[i], np.dot(w, samples[i])) < 1:
            w = np.multiply((1 - eta_t), w) + np.multiply(eta_t*C*labels[i], samples[i])
    return w

if __name__ == '__main__':
    main(sys.argv[1:])