from numpy import *
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from numpy import linalg as LA

def main(args):
    mnist = fetch_mldata('MNIST original')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = 0,8
    train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
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

def perceptron(normalized_data, data_labels):
    vector_size = len(normalized_data[0])
    samples_num = len(normalized_data)
    w = np.zeros(vector_size, dtype=float)
    # initialize norm in order to calculate without division by 0
    w_norm = 1
    for i in range(samples_num):
        correct_label = data_labels[i]
        calc = np.dot(w, normalized_data[i])/w_norm
        predication = 0 if calc >= 0 else 8
        if predication == correct_label:
            continue
        one_loss_prediction = 1 if correct_label == 0 else -1
        w = w + np.multiply(one_loss_prediction, normalized_data[i])
        w_norm = LA.norm(w)
    return w

if __name__ == '__main__':
    main(sys.argv[1:])