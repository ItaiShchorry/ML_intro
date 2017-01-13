import sys
from sklearn.datasets import fetch_mldata
import numpy.random
import math
import numpy as np
import operator
import matplotlib.pyplot as plt
from numpy import linalg as LA
import sklearn.preprocessing
from numpy import linalg as LA
import pandas as pd
from scipy import stats

def main():
    #a = [1, 2, 3, 4, 5]
    #aa = pd.DataFrame({'a':[1,2,np.nan], 'b':[np.nan,1,np.nan], 'c':[5,6,7], 'd':[np.nan,np.nan,np.nan]})
    # print ('shape ',aa.shape[0] * 2 / 2)
    # print ('shape div ',aa.shape[0] * 1 / 2)
    # print ('shape div ', aa.shape[0] * 0.5)
    # print ('aa: ', aa)
    # aa = aa.dropna(1,thresh= aa.shape[0] * 2 / 3)
    # print ('aa new', aa)
    # aa_val = aa.isnull().sum()
    # aa_len = aa.shape[0]
    #aa_val2 = np.divide(np.array(aa_val),aa_len)

    x1 = [-1, -1, 0, 0, 1, 1]
    x2 = [1, 1, 0, 0, -1, -1]

    h = lambda x: 1. if x[0] < 0 else -1.
    print ("hhh", h(x1))
    print ("hhh", h(x2))

    options = [0,1]
    thresholds = [0]
    pixels = [0,1,2]
    h_arr = [(i,j) for i in thresholds for j in pixels]
    h_array = np.array([])
    #h_array = np.append(h_array, lambda x, k=j, m=i: 1. if x[k] < i else -1.)
    #for i in thresholds:
     #   for j in pixels:
      #      h_array = np.append(h_array, lambda x: 1. if (x[j] < i) else -1.)
       #     h_array = np.append(h_array, lambda x: -1. if (x[j] < i) else 1.)
        #    h_array = np.append(h_array, lambda x, k=j, m=i: 1. if x[k] < i else -1.)
            #h = lambda x: pos_under(x, j, i)
            #h_array = np.append(h_array,h)
            #h = lambda x: pos_over(x, j, i)
            #h_array = np.append(h_array,h)
    #for i in range(len(h_array)):
    #    print ("h", i, " x1:", h_array[i](x1))
    #    print ("h", i, " x2:", h_array[i](x2))
    print (hypothesys(x1, (0,0,0)))
    print (hypothesys(x2, (0,0,0)))
    print (hypothesys([-1,0,0], (0,0,0)))
    print (hypothesys([-0,0,0], (0,0,0)))
    print (hypothesys([1,0,0], (0,0,0)))
    print (hypothesys([2,0,0], (0,0,0)))
    print (hypothesys([3,0,0], (0,0,0)))


def hypothesys(x, (i,j,k)):
    if k == 0:
        return pos_under(x, (i,j))
    else:
        return pos_over(x, (i, j))

def pos_under(x, (i,j)):
    if x[j] <= i:
        return (1.)
    else:
        return (-1.)

def pos_over(x, (i,j)):
    if x[j] <= i:
        return (-1.)
    else:
        return (1.)
    #rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2))
   # print (h)
    #stats.ttest_1samp()

if __name__ == '__main__':
    sys.exit(main())