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

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos)*2-1

#validation_data_unscaled = data[train_idx[6000:], :].astype(float)
#validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000+test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000+test_idx[:test_data_size]] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
#validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

samples_num = train_data.shape[0]
pixels_num = train_data.shape[1]
sorted_pixels = [[1. for i in range(samples_num)] for j in range(pixels_num)]
sorted_pixel_labels = [[1. for i in range(samples_num)] for j in range(pixels_num)]
idx = [[1. for i in range(samples_num)] for j in range(pixels_num)]

#preprocessing phase for weak learners
for pixel in range(pixels_num):
    idx[pixel] = np.argsort([train_data[sample_index][pixel] for sample_index in range(samples_num)])
    sorted_pixels[pixel] = [train_data[j][pixel] for j in idx[pixel]]
    sorted_pixel_labels[pixel] = [train_labels[j] for j in idx[pixel]]

def main(args):
    # output path:
    if len(args) == 1:
        output = args[0] + '/'
        if not os.path.exists(output):
            print("Path does not exist!")
            sys.exit(2)
    elif len(args) > 1:
        print("usage: Q5.py <output_path>")
        sys.exit(2)
    else:
        output = ''

    # Section A
    D = np.array([(1. / samples_num) for i in range(samples_num)])
    T=50
    H = []
    alphas = []
    train_error = []
    test_error = []
    train_lossFunc = []
    test_lossFunc = []
    t_array = [i for i in range (1,T+1)]
    for t in range (T):
        (h,train_error_i,alpha,D) = set_params(D)
        H.append(h)
        alphas.append(alpha)
        train_error.append(test_H(H,alphas,t,True))
        test_error.append(test_H(H,alphas,t,False))
        train_lossFunc.append(calcLossFunc(H, alphas, t, True))
        test_lossFunc.append(calcLossFunc(H, alphas, t, False))
        print ("iteration ", t + 1 ," train error ",train_error[t]," test error ",test_error[t])
        print ("iteration ", t + 1, " train lossFunc ", train_lossFunc[t], " test lossFunc ", test_lossFunc[t])


    #plots
    plt.figure(1)
    plt.plot(t_array, train_error)
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('Training error by iterations number')
    img_save = output + 'training error'
    plt.savefig(img_save)

    plt.figure(2)
    plt.plot(t_array, test_error)
    plt.xlabel('t')
    plt.ylabel('error')
    plt.title('Test error by iterations number')
    img_save = output + 'test error'
    plt.savefig(img_save)

    plt.figure(3)
    plt.plot(t_array, train_lossFunc)
    plt.xlabel('t')
    plt.ylabel('loss function')
    plt.title('Training loss function by iterations number')
    img_save = output + 'training loss function'
    plt.savefig(img_save)

    plt.figure(4)
    plt.plot(t_array, test_lossFunc)
    plt.xlabel('t')
    plt.ylabel('loss function')
    plt.title('Test loss function by iterations number')
    img_save = output + 'test loss function'
    plt.savefig(img_save)


def set_params(D):
    #h are saved as (threshold,pixel,option)
    best_h = (0,0,0)
    best_error = 1

    for pixel in range(pixels_num):
        curr_threshold = sorted_pixels[pixel][0]
        curr_estimator_plus = (curr_threshold, pixel, 0)#if pixel <= curr_threshold predict 1 (if > 0 predict -1)
        curr_estimator_minus = (curr_threshold, pixel, 1)#if pixel <= curr_threshold predict -1 (if > 0 predict 1)
        curr_error_plus = test_h((curr_estimator_plus),D)
        curr_error_minus = test_h((curr_estimator_minus),D)
        thresh_idx = 1
        while ((thresh_idx < samples_num) and (sorted_pixels[pixel][thresh_idx] == curr_threshold)):
            thresh_idx += 1
        while thresh_idx<samples_num:# the sample pixels are the threholds
            curr_threshold = sorted_pixels[pixel][thresh_idx]
            curr_estimator_plus = (curr_threshold, pixel, 0)
            curr_estimator_minus = (curr_threshold, pixel, 1)
            curr_error_plus -= sorted_pixel_labels[pixel][thresh_idx] * D[idx[pixel][thresh_idx]]
            curr_error_minus += sorted_pixel_labels[pixel][thresh_idx] * D[idx[pixel][thresh_idx]]
            thresh_idx+=1
            #deal with consecutive sample pixels with same value
            while ((thresh_idx < samples_num) and not (sorted_pixels[pixel][thresh_idx] > curr_threshold)):
                curr_error_plus -= sorted_pixel_labels[pixel][thresh_idx] * D[idx[pixel][thresh_idx]]
                curr_error_minus += sorted_pixel_labels[pixel][thresh_idx] * D[idx[pixel][thresh_idx]]
                thresh_idx+=1
            # Peek best h
            if best_error > curr_error_plus:
                best_error = curr_error_plus
                best_h = curr_estimator_plus
            if best_error > curr_error_minus:
                best_error = curr_error_minus
                best_h = curr_estimator_minus
    # Update D
    alpha = 0.5 * math.log((1. -best_error)/best_error)
    D = np.array([D[i]*math.exp(-alpha) if train_labels[i] == hypothesys(train_data[i],best_h)
                  else D[i]*math.exp(alpha) for i in range (samples_num)])
    D = np.array([d / D.sum() for d in D])
    return (best_h,best_error,alpha,D)


def test_h(h,D):
    return (np.dot(np.array([hypothesys(train_data[i],h) != train_labels[i] for i in range (samples_num)]),D))

# returns the error
# the probability of predicting 0 is 0 so we can use the sign function
def test_H(H,alphas,T,is_train):
    data = train_data if is_train else test_data
    labels = train_labels if is_train else test_labels
    len = train_data.shape[0] if is_train else test_data.shape[0]
    y_hat = np.array([])
    for i in range(len):
        coef = 0
        for t in range (T+1):
            coef+=alphas[t]*hypothesys(data[i],H[t])
        y_hat = np.append(y_hat,sign(coef))
    return (np.array([y_hat[i] != labels[i] for i in range(len)]).mean())

def calcLossFunc(H,alphas,T,is_train):
    data = train_data if is_train else test_data
    labels = train_labels if is_train else test_labels
    len = train_data.shape[0] if is_train else test_data.shape[0]
    losses = []
    for i in range (len):
        coef = 0
        for t in range (T+1):
            coef += alphas[t] * hypothesys(data[i],H[t])
        losses.append(math.exp(-labels[i]*coef))
    return np.array(losses).mean()

def hypothesys(x, h):
    if h[2] == 0:
         return pos_under(x, h)
    else:
         return pos_over(x, h)

def pos_under(x, h):
    if x[h[1]] <= h[0]:
        return (1.)
    else:
        return (-1.)

def pos_over(x, h):
    if x[h[1]] <= h[0]:
        return (-1.)
    else:
        return (1.)

if __name__ == '__main__':
    main(sys.argv[1:])
