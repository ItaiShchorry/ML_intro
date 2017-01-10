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

m = train_data.shape[0]
p = train_data.shape[1]

def main(args):
    D = np.array([(1/m) for i in range(m)])
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
    ttrain_data = np.array(train_data)
    #max_pixel = np.max(ttrain_data)
    #min_pixel = np.min(ttrain_data)
    #thresholds = [i for i in range(min_pixel.astype(int), max_pixel.astype(int))]
    ###thresholds = [i for i in range(-12,12,4)]
    thresholds = [-128, -5, 0, 5, 128]
    pixels = [j for j in range(200,(p-200))]
    #pixels = (j for j in range(p))
    #h_pos_array = np.array([(lambda x: sign(x[j]-thresholds[i])) for i in thresholds for j in pixels])
    #h_neg_array = np.array([(lambda x: -sign(x[j]-thresholds[i])) for i in thresholds for j in pixels])
    #h_array = np.concatenate((h_pos_array,h_neg_array))
    h_array = []
    for i in thresholds:
        for j in pixels:
            h_array.append(lambda x: 1. if x[j] <= i else -1.)
            h_array.append(lambda x: -1. if x[j] <= i else 1.)
    h_array = np.array(h_array)
    print ("num of h: ",h_array.shape)
    D = np.array([(1. / m) for i in range(m)])
    for k in range (0,100,5):
        for l in range (0, 3000, 500):
            print ("h ", l ," result ", h_array[l](train_data[k]))
    return
    print (D[1:5])
    T=50
    H = []
    alphas = []
    train_error = []
    test_error = []
    train_lossFunc = []
    test_lossFunc = []
    t_array = [i for i in range (1,T+1)]
    for t in range (T):
        (h,train_error_i,alpha,D) = set_params(D,h_array)
        H.append(h)
        alphas.append(alpha)
        train_error.append(test_H(H,alphas,t,True))
        test_error.append(test_H(H,alphas,t,False))
        train_lossFunc.append(calcLossFunc(H, alphas, t, True))
        test_lossFunc.append(calcLossFunc(H, alphas, t, False))
        print ("iteration ", t+1 ," train error ",train_error[t]," test error ",test_error[t])
        print ("iteration ", t + 1, " train lossFunc ", train_lossFunc[t], " test lossFunc ", test_lossFunc[t])

    print ("train error ",train_error)
    print ("test error ", test_error)
    print ("train lossFunc ", train_lossFunc)
    print ("test lossFunc ", test_lossFunc)

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

# each hypothesys will be represented by [threshold, sign (for above it), j (pixel)]
# we will use each pixel values as threshold
def set_params(D,h_array):
    best_h = h_array[0]
    best_error = 1.
    for h in h_array:
        curr_error = test_h(h,D)
        if curr_error<best_error:
            print ("new best error ", curr_error)
            best_error = curr_error
            best_h = h
    print ("############")
    print ("best error", best_error)
    print ("(1. -best_error)/best_error", (1. -best_error)/best_error)
    print ("math.log((1. -best_error)/best_error) 1 = ", math.log((1. -best_error)/best_error))
    b = (1. -best_error)/best_error
    print ("b", b)
    print ("math.log((1. -best_error)/best_error) 2 = ", math.log(b))
    alpha = 0.5 * math.log((1. -best_error)/best_error)
    D = np.array([D[i]*math.exp(-alpha) if train_labels[i] == best_h(train_data[i])
                  else D[i]*math.exp(alpha) for i in range (m)])
    D = [d / D.sum() for d in D]
    print ("best error", best_error," alpha ", alpha," D[1:20]", D[1:10])
    return (best_h,best_error,alpha,D)


def test_h(h,D):
    return (np.dot(np.array([h(train_data[i]) != train_labels[i] for i in range (m)]),np.array(D)))

#returns the error
def test_H(H,alphas,T,is_train):
    data = train_data if is_train else test_data
    labels = train_labels if is_train else test_labels
    len = train_data.shape[0] if is_train else test_data.shape[0]
    y_hat = [sign(np.array(alphas[t]*H[t](data[i]) for t in range(T)).sum()) for i in range (len)]
    return (np.array([y_hat[i] != labels[i] for i in range(len)]).mean())

def calcLossFunc(H,alphas,T,is_train):
    data = train_data if is_train else test_data
    labels = train_labels if is_train else test_labels
    len = train_data.shape[0] if is_train else test_data.shape[0]
    coef = 0
    losses = []
    for i in range (len):
        for t in range (T):
            coef += alphas[t] * H[t](data[i])
        #coef = coef*labels[i]*(-1.)
        #coef = math.exp(-labels[i]*coef)
        losses.append(math.exp(-labels[i]*coef))
    return np.array(losses).mean()
    #return ( np.array([math.exp(-labels[i]*np.array((alphas[t] * H[t](data[i])) for t in range(T)).sum()) for i in range(len)]).mean() )

if __name__ == '__main__':
    main(sys.argv[1:])
