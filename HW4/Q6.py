import sys
import os
from numpy import *
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
from scipy import linalg as LA
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos) * 2 - 1

# validation_data_unscaled = data[train_idx[6000:], :].astype(float)
# validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000 + test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000 + test_idx[:test_data_size]] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
# validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

m = train_data.shape[0]
p = train_data.shape[1]
eigan_values_number = 100
eigan_vectors_number = 5


def main(args):
    D = np.array([(1 / m) for i in range(m)])
    # output path:
    if len(args) == 1:
        output = args[0] + '/'
        if not os.path.exists(output):
            print("Path does not exist!")
            sys.exit(2)
    elif len(args) > 1:
        print("usage: Q5_not_relevant.py <output_path>")
        sys.exit(2)
    else:
        output = ''

        # section A
    samples_8 = train_data[np.where(train_labels == 1)]
    (eigan_values, eigan_vectors) = PCA(samples_8, 100)
    mean_vector = np.array(samples_8).mean(axis=0)
    output_plots_A_B_C(eigan_values, mean_vector, eigan_vectors, output, 'a')

    # section B
    samples_0 = train_data[np.where(train_labels == -1)]
    (eigan_values, eigan_vectors) = PCA(samples_0, 100)
    mean_vector = np.array(samples_0).mean(axis=0)
    output_plots_A_B_C(eigan_values, mean_vector, eigan_vectors, output, 'b')

    # section C
    (eigan_values, eigan_vectors) = PCA(train_data, 100)
    mean_vector = np.array(train_data).mean(axis=0)
    output_plots_A_B_C(eigan_values, mean_vector, eigan_vectors, output, 'c')

    # section D
    vec_1_samples_8 = [np.dot(eigan_vectors[0], sample_8) for sample_8 in samples_8]
    vec_2_samples_8 = [np.dot(eigan_vectors[1], sample_8) for sample_8 in samples_8]
    vec_1_samples_0 = [np.dot(eigan_vectors[0], sample_0) for sample_0 in samples_0]
    vec_2_samples_0 = [np.dot(eigan_vectors[1], sample_0) for sample_0 in samples_0]
    plt.figure(7)
    plt.plot(vec_1_samples_8, vec_2_samples_8, 'ro', vec_1_samples_0, vec_2_samples_0, 'bo')
    plt.title("2d projections. 8 - red, 0 - blue")
    plt.xlabel("first vector")
    plt.ylabel("second vector")
    img_save = output + "Q6_section_d_2D_projections"
    plt.savefig(img_save)

    # section E
    posFig1 = samples_8[0]
    posFig2 = samples_8[1]
    negFig1 = samples_0[0]
    negFig2 = samples_0[1]
    images = np.array([samples_8[0], samples_8[1], samples_0[0], samples_0[1]])
    X_hat_10 = np.matmul(np.matmul(eigan_vectors[:10].T, eigan_vectors[:10]), images.T).T
    X_hat_30 = np.matmul(np.matmul(eigan_vectors[:30].T, eigan_vectors[:30]), images.T).T
    X_hat_50 = np.matmul(np.matmul(eigan_vectors[:50].T, eigan_vectors[:50]), images.T).T
    output_plots_E([images, X_hat_10, X_hat_30, X_hat_50], output)


def PCA(data, dims_rescaled_data=2):
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = LA.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evals = evals[:dims_rescaled_data]
    evecs = evecs[:, :dims_rescaled_data].T
    return evals, evecs


def output_plots_A_B_C(eigan_values, mean_vector, eigan_vectors, output, section):
    if (section == 'a'):
        fig_num = 1
        head = '8'
    elif (section == 'b'):
        fig_num = 3
        head = '0'
    elif (section == 'c'):
        fig_num = 5
        head = '0_&_8'

    # plotting vectors
    plt.figure(fig_num)
    plt.subplot(231)
    plt.title(head + ' mean vector')
    plt.imshow(reshape(mean_vector, (28, 28)), cmap='gray', interpolation='nearest')
    for i in range(232, 237):
        plt.subplot(i)
        plt.title(head + ' eigan vector %d' % (i - 231))
        plt.imshow(reshape(eigan_vectors[i - 232], (28, 28)), cmap='gray', interpolation='nearest')
    img_save = output + "Q6_section_" + section + "_" + head + "_eigan_vectors"
    plt.savefig(img_save)

    # plotting eigan values
    dimensions_array = [i for i in range(1, eigan_values_number + 1)]
    plt.figure(fig_num + 1)
    plt.xlabel('dimension')
    plt.ylabel('eigan_value')
    plt.title('Eigan values of ' + head + ' samples by dimensions')
    plt.plot(dimensions_array, eigan_values)
    img_save = output + "Q6_section_" + section + "_" + head + "_eigan_values"
    plt.savefig(img_save)


def output_plots_E(image_constructions, output):
    plt.figure(8)
    title_array = ["original", "k=10", "k=30", "k=50"]
    image_headers = ["pos image 1", "pos image 2", "neg image 1", "neg image 2"]
    for image_index in range(4):
        for i in range(4):
            plt.subplot2grid((4, 4), (image_index, i))
            if (image_index == 0):
                plt.title(title_array[i])
            if (i == 0):
                plt.ylabel(image_headers[image_index])
            plt.imshow(reshape(image_constructions[i][image_index], (28, 28)), cmap='gray', interpolation='nearest')
    img_save = output + "Q6_section_e_constructions"
    plt.savefig(img_save)


if __name__ == '__main__':
    main(sys.argv[1:])
