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
    points = generatePoints(100)
    plt.plot(points[0], points[1], 'ro')
    plt.axis([0, 1, -0., 1.1])
    ax = plt.gca()
    l1 = lns.Line2D((0.25, 0.25), (-0.1,1.1), ls='--')
    l2 = lns.Line2D((0.5, 0.5), (-0.1,1.1), ls='--')
    l3 = lns.Line2D((0.75, 0.75), (-0.1,1.1), ls='--')
    ax.add_line(l1)
    ax.add_line(l2)
    ax.add_line(l3)
    #plt.show()
    k = 10
    intervals, besterror = find_best_interval(points[0], points[1], k)
    for i in range(k):
        l = lns.Line2D((intervals[i][0], intervals[i][1]), (0.05, 0.05), ls='-')
        ax.add_line(l)
    plt.show()

def generatePoints(num_of_points):
    x = np.random.uniform(0, 1, num_of_points)
    x = np.sort(x)
    n = 1
    y = np.array([np.random.binomial(n, 0.8) if (xi >= 0 and xi <= 0.25) or (xi >= 0.5 and xi <= 0.75) else np.random.binomial(n, 0.1) for xi in x])
    return [x, y]

if __name__ == '__main__':
    sys.exit(main())
