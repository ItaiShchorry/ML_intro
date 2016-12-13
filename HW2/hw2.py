import sys
from numpy import *
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt

def main(args):
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0,8
    train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])# TODO why is the [0] at the end?
    test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos)*2-1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos)*2-1

    test_data_unscaled = data[60000+test_idx, :].astype(float)
    test_labels = (labels[60000+test_idx] == pos)*2-1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

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

    perceptron_experiments(train_data, train_labels, test_data, test_labels)

def perceptron_experiments(train_data, train_labels, test_data, test_labels):
    total_samples_num = len(train_data)
    vector_size = len(train_data[0])
    train_idx = np.arange(total_samples_num)
    total_normalized_data = np.array([(sample/LA.norm(sample)) for sample in train_data])
    neg, pos = -1, 1
    num_of_experiments = 100
    samples_num=[5,10,50,100,500,1000,5000]
    experiment_accuracy = np.zeros(num_of_experiments, dtype=float)
    sample_num_accuracy = np.zeros(21, dtype=float).reshape(7,3) # samples_num, mean, 5%, 95% for every row
    for i in range(7):
        for j in range(num_of_experiments):
            train_idx = numpy.random.RandomState(j).permutation(train_idx)
            sub_train_data = np.array(total_normalized_data[train_idx[:samples_num[i]], :])
            sub_train_labels = np.array(train_labels[train_idx[:samples_num[i]]])
            w = perceptron(sub_train_data, sub_train_labels, samples_num[i], vector_size)
            experiment_accuracy[j] = np.mean([test_perceptron(w,d,l) for (d,l) in zip(test_data, test_labels)])
        print ("done ",i) #TODO
        experiment_accuracy = np.sort(experiment_accuracy)
        #sample_num_accuracy[i][0] = numpy.round(samples_num[i])
        sample_num_accuracy[i][0] = np.mean(experiment_accuracy)
        sample_num_accuracy[i][1] = experiment_accuracy[4] # 5% percentile
        sample_num_accuracy[i][2] = experiment_accuracy[94] # 95% percentile

    #TODO print sample_num_accuracy in table
    print (sample_num_accuracy)
    #section b
    w = perceptron(total_normalized_data, train_labels, total_samples_num, vector_size)
    w_show = sklearn.preprocessing.minmax_scale(w,feature_range=(0,255))
    plt.figure(1)
    plt.imshow(reshape(w_show,(28,28)),interpolation = 'nearest', cmap = 'gray')
    plt.title('weight vector')

    #section c
    predictions_grade = np.array([test_perceptron(w,d,l) for (d,l) in zip(test_data, test_labels)])
    print ("fully trained perceptron accuracy is: ", np.mean(predictions_grade))

    #section d
    test_idx = np.arange(len(test_data))
    wrong_sample_idx = numpy.random.choice(test_idx[np.where(predictions_grade[test_idx] == 0)])
    wrong_sample = sklearn.preprocessing.minmax_scale(test_data[wrong_sample_idx], feature_range=(0, 255))
    if test_labels[wrong_sample_idx] == 1:
        true_digit, false_digit = 8, 0
    else:
        true_digit, false_digit = 0, 8
    plt.figure(2)
    plt.imshow(reshape(wrong_sample, (28, 28)), interpolation='nearest', cmap = 'gray')
    plt.title('bad sample. digit is %d while prediction is %d' %(true_digit ,false_digit))
    plt.show()

def test_perceptron(w, sample, data_label):
    prediction = 1 if np.dot(w, sample) >= 0 else -1
    return (1.0 if prediction == data_label else 0.0)

def perceptron(normalized_data, data_labels, samples_num, vector_size):
    w = np.zeros(vector_size, dtype=float)
    for i in range(samples_num):
        prediction = 1 if np.dot(w, normalized_data[i])>=0 else -1
        if prediction != data_labels[i]:
            w = w + np.multiply(data_labels[i], normalized_data[i])
            w = w/LA.norm(w)
    return w

if __name__ == '__main__':
    main(sys.argv[1:])