from numpy import *
import sys
from sklearn.datasets import fetch_mldata
import numpy.random
import math
import numpy as np
import operator
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as lns

def find_best_interval(xs, ys, k):
    assert all(array(xs) == array(sorted(xs))), "xs must be sorted!"
    
    m = len(xs)
    P = [[None for j in range(k+1)] for i in range(m+1)]
    E = zeros((m+1, k+1), dtype=int)
    
    # Calculate the cumulative sum of ys, to be used later
    cy = concatenate([[0], cumsum(ys)])
    
    # Initialize boundaries:
    # The error of no intervals, for the first i points
    E[:m+1,0] = cy
    
    # The minimal error of j intervals on 0 points - always 0. No update needed.        
        
    # Fill middle
    for i in range(1, m+1):
        for j in range(1, k+1):
            # The minimal error of j intervals on the first i points:
            
            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0,i+1):  
                next_errors = E[l,j-1] + (cy[i]-cy[l]) + concatenate([[0], cumsum((-1)**(ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l,i+1)[min_error])))

            E[i,j], P[i][j] = min(options)
    
    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k,0,-1):
        best.append(cur)
        cur = P[cur[0]][i-1]       
        if cur == None:
            break 
    best = sorted(best)
    besterror = E[m,k]
    
    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:]+exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l,u in best]
    
    return intervals, besterror

def main():
    #section a
    points = generatePoints(100)
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('section a')
    plt.plot(points[0], points[1], 'ro')
    plt.axis([0, 1, -0.1, 1.1])
    plt.axvline(0.25, -0.1,1.1,color = 'b')
    plt.axvline(0.5, -0.1, 1.1, color='b')
    plt.axvline(0.75, -0.1, 1.1, color='b')

    k = 2
    intervals, besterror = find_best_interval(points[0], points[1], k)
    #for i in range(k):
    #    l = lns.Line2D((intervals[i][0], intervals[i][1]), (0.05, 0.05), ls='-')
    #    ax.add_line(l)

    plt.axvline(intervals[0][0], -0.1, 1.1, color = 'r')
    plt.axvline(intervals[0][1], -0.1, 1.1, color = 'r')
    plt.axvline(intervals[1][0], -0.1, 1.1, color = 'r')
    plt.axvline(intervals[1][1], -0.1, 1.1, color = 'r')
    plt.axhline(1.05, intervals[0][0], intervals[0][1], color = 'r')
    plt.axhline(1.05, intervals[1][0], intervals[1][1], color='r')


    #section c
    k=2
    m=[i for i in range(10,25,5)] #TODO change the high limit from 25 to 105
    T=100
    experiments = np.array([[0.1 for i in range (T)],[0.1 for i in range (T)]]) # true error, empirical error, m[i]
    Cplot_array = np.array([[0.1 for i in range (len(m))],[0.1 for i in range (len(m))]])
    for i in range (len(m)):
        for j in range (T):
            points = generatePoints(m[i])
            intervals, besterror = find_best_interval(points[0], points[1], k)
            experiments[0][j] = calcTrueError(intervals)
            experiments[1][j] = calcEmpiricalError(intervals, points)
        print ("mean for: ",m[i]," true error: ",experiments[0][i]," empirical error: ",experiments[1][j]) #TODO
        (Cplot_array[0][i]), Cplot_array[1][i] = np.mean(experiments, axis = 1)

    plt.figure(2)
    plt.plot(m,Cplot_array[0],'r',m,Cplot_array[1],'b')
    plt.title('section c: red - true error, blue - empirical error')
    plt.xlabel('samples number')
    plt.ylabel('error')

    #section D
    max_k = 20
    k_array = [i for i in range(1,max_k+1)]
    error_array = [0.0 for i in range(max_k)]
    m = 50
    points = generatePoints(m)
    for i in range(max_k):
        intervals, besterror = find_best_interval(points[0], points[1], k_array[i])
        error_array[i] = calcEmpiricalError(intervals,points)

    plt.figure(3)
    plt.plot(k_array,error_array)
    plt.title('section d')
    plt.xlabel('k')
    plt.ylabel('error')

    #section e
    T = 10 #TODO change to 100
    experiments = np.array([[0.1 for i in range(T)], [0.1 for i in range(T)]])  # true error, empirical error, m[i]
    Eplot_array = np.array([[0.1 for i in range(1,max_k+1)], [0.1 for i in range(1,max_k+1)]])
    for i in range(max_k):
        for j in range(T):
            points = generatePoints(m)
            intervals, besterror = find_best_interval(points[0], points[1], k_array[i])
            experiments[0][j] = calcTrueError(intervals)
            experiments[1][j] = calcEmpiricalError(intervals, points)
        Eplot_array[0][i], Eplot_array[1][i] = np.mean(experiments, axis=1)
        print ("mean for k=: ", k_array[i], " true error: ", Eplot_array[0][i], " empirical error: ", Eplot_array[1][i])  # TODO

    plt.figure(4)
    plt.plot(k_array,Eplot_array[0],'r',k_array,Eplot_array[1],'b')
    plt.title('section e: red - true error, blue - empirical error')
    plt.xlabel('k')
    plt.ylabel('error')


    #section f
    #choosing 5-fold cross validation for determining k
    k_fold = 10 #TODO change to 5
    indexes = np.array([i for i in range(m)])
    random.shuffle(indexes)
    error_array = [i for i in range (max_k)]
    for k in range (1,21,5): #TODO remove the 5 step
        comulative_error = 0.0
        for i in range (0,50,k_fold):
            train = [[points[0] for j in range(m) if j not in indexes[i:i+k_fold]],[points[1] for j in range(m) if j not in indexes[i:i+k_fold]]]
            test = [[points[0] for j in indexes[i:i+k_fold]],[points[1] for j in indexes[i:i+k_fold]]]
            intervals, besterror = find_best_interval(points[0], points[1], k)
            comulative_error += calcEmpiricalError(intervals, points)
        error_array[1][k-1] = comulative_error / 10

    plt.figure(5)
    plt.title('section f: 5-fold cross validation')
    plt.plot(k_array,error_array)
    plt.xlabel('k')
    plt.ylabel('averaged error')
    plt.show() #TODO


def generatePoints(m):
    x = np.random.uniform(0, 1, m)
    x = np.sort(x)
    n = 1
    y = np.array([np.random.binomial(n, 0.8) if (xi >= 0 and xi <= 0.25) or (xi >= 0.5 and xi <= 0.75) else np.random.binomial(n, 0.1) for xi in x])
    return [x, y]

#Section b
def calcTrueError(intervals):
    num_of_intervals = len(intervals)
    flattend_intervals = [0.1 for i in range (2*num_of_intervals + 2)]
    flattend_intervals[0] = 0.0
    for i in range(num_of_intervals):
        flattend_intervals[2*i + 1] = intervals[i][0]
        flattend_intervals[2*i + 2] = intervals[i][1]
    flattend_intervals[-1] = 1.0

    true_error = 0
    i = 0
    # calcultaing true errors. i % 2 == 1: right side of an interval. i % 2 == 0: left side
    # 0-0.25
    while (flattend_intervals[i]<0.25):
        i += 1
        error_possibility = 0.8 if (i % 2 == 1) else 0.2
        true_error += error_possibility * ( min(flattend_intervals[i],0.25)-flattend_intervals[i - 1])

    # 0.25-0.5
    error_possibility = 0.1 if (i % 2 == 1) else 0.9
    true_error += error_possibility * (min(flattend_intervals[i],0.5) - 0.25)
    while (flattend_intervals[i] < 0.5): #
        i += 1
        error_possibility = 0.1 if (i % 2 == 1) else 0.9
        true_error += error_possibility * (min(flattend_intervals[i], 0.5) - flattend_intervals[i - 1])

    # 0.5-0.75
    error_possibility = 0.8 if (i % 2 == 1) else 0.2
    true_error += error_possibility * (min(flattend_intervals[i], 0.75) - 0.5)
    while (flattend_intervals[i] < 0.75):
        i += 1
        error_possibility = 0.8 if (i % 2 == 1) else 0.2
        true_error += error_possibility * (min(flattend_intervals[i], 0.75) - flattend_intervals[i - 1])

    #0.75-1
    error_possibility = 0.1 if (i % 2 == 1) else 0.9
    true_error += error_possibility * (min(flattend_intervals[i], 1) - 0.75)
    while (flattend_intervals[i] < 1):  #
        i += 1
        error_possibility = 0.1 if (i % 2 == 1) else 0.9
        true_error += error_possibility * (min(flattend_intervals[i], 1) - flattend_intervals[i - 1])

    return true_error


def calcEmpiricalError(intervals,points):
    num_of_intervals = len(intervals)
    flattend_intervals = [0.1 for i in range(2 * num_of_intervals + 2)]
    flattend_intervals[0] = 0.0
    for i in range(num_of_intervals):
        flattend_intervals[2 * i + 1] = intervals[i][0]
        flattend_intervals[2 * i + 2] = intervals[i][1]
    flattend_intervals[-1] = 1.0

    interval_index = 0
    sample_index = 0
    len_f = len(flattend_intervals)
    m = len(points[0])
    count = 0

    while (interval_index < len_f - 2 and sample_index < m):
        while (points[0][sample_index] > flattend_intervals[interval_index + 1]):
            interval_index += 1
        estimation = interval_index % 2
        if estimation != points[1][sample_index]:
            count += 1
        sample_index += 1

    return (float(count) / m)



if __name__ == '__main__':
    sys.exit(main())
