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
    a = [1, 2, 3, 4, 5]
    b = [1, 4, 9, 16, 25]
    plt.figure(1)
    plt.title('hhhhhhhhhhhhhhhhhhhhhhhhhh')
    i = 231
    plt.subplot(i)
    plt.plot(a,b)
    plt.title ('1')
    plt.subplot(232)
    plt.plot(a, b)
    plt.title('2')
    plt.subplot(233)
    plt.plot(a, b)
    plt.title('3')
    plt.subplot(234)
    plt.plot(a, b)
    plt.title('4')
    plt.subplot(235)
    plt.plot(a, b)
    plt.title('5')
    plt.subplot(236)
    plt.plot(a, b)
    plt.title('6')
    plt.show()
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

    for i in range (0):
        print ("hi")
    #rvs = stats.norm.rvs(loc=5, scale=10, size=(50, 2))
   # print (h)
    #stats.ttest_1samp()

if __name__ == '__main__':
    sys.exit(main())