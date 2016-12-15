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
        print("usage: Q2.py <output_path>")
        sys.exit(2)
    else:
        output = ''
    # Section A
    C_list = [math.pow(10, x) for x in range(-10, 11)]
    best_accuracy = 0
    best_C = 0
    for c in C_list:
        clf = svm.LinearSVC(C=c, loss='hinge', fit_intercept=False)
        clf.fit(train_data, train_labels)
        prediction = clf.predict(validation_data)
        accuracy = 1.0 * np.sum(np.array(
            [1.0 if prediction[i] == validation_labels[i] else 0.0 for i in range(validation_labels.shape[0])])) / \
                   validation_labels.shape[0]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = c
    print('We can see that the best C value for validation set in the given range is ', best_C)
    print('Trying now the range: from 10^-8 to 10^-6 steps of 5*10^-8')

    C_list = [x for x in np.arange(math.pow(10, -8), math.pow(10, -6) + math.pow(10, -8), math.pow(10, -8))]
    best_accuracy = 0
    best_C = 0
    validation_set_accuracies = []
    training_set_accuracies = []
    for c in C_list:
        clf = svm.LinearSVC(C=c, loss='hinge', fit_intercept=False)
        clf.fit(train_data, train_labels)
        prediction_for_validation = clf.predict(validation_data)
        accuracy_for_validation = 1.0 * np.sum(np.array(
            [1.0 if prediction_for_validation[i] == validation_labels[i] else 0.0 for i in
             range(validation_labels.shape[0])])) / validation_labels.shape[0]
        prediction_for_training = clf.predict(train_data)
        accuracy_for_training = 1.0 * np.sum(np.array(
            [1.0 if prediction_for_training[i] == train_labels[i] else 0.0 for i in range(train_labels.shape[0])])) / \
                                train_labels.shape[0]
        validation_set_accuracies.append(accuracy_for_validation)
        training_set_accuracies.append(accuracy_for_training)
        if accuracy_for_validation > best_accuracy:
            best_accuracy = accuracy_for_validation
            best_C = c

    print('We can see that the best C value for validation set in the given range is ', best_C)

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

    # Section C
    clf = svm.LinearSVC(C=best_C, loss='hinge', fit_intercept=False)
    clf.fit(train_data, train_labels)
    plt.figure(2)
    plt.imshow(reshape(clf.coef_, (28, 28)), interpolation='nearest', cmap='gray')
    plt.title('weight vector')
    img_save = output + 'Q2_weight_vector'
    plt.savefig(img_save)
    prediction_for_test = clf.predict(test_data)

    # Section D
    accuracy_for_training = 1.0 * np.array(\
        [1.0 if prediction_for_test[i] == test_labels[i] else 0.0 for i in range(test_labels.shape[0])]).sum() / \
                            test_labels.shape[0]
    print('The accuracy of the linear SVM with the best C on the test set is: ', accuracy_for_training)


if __name__ == '__main__':
    main(sys.argv[1:])
