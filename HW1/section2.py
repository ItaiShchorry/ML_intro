from numpy import *
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def find_best_interval(xs, ys, k):
    assert all(array(xs) == array(sorted(xs))), "xs must be sorted!"

    m = len(xs)
    P = [[None for j in range(k + 1)] for i in range(m + 1)]
    E = zeros((m + 1, k + 1), dtype=int)

    # Calculate the cumulative sum of ys, to be used later
    cy = concatenate([[0], cumsum(ys)])

    # Initialize boundaries:
    # The error of no intervals, for the first i points
    E[:m + 1, 0] = cy

    # The minimal error of j intervals on 0 points - always 0. No update needed.        

    # Fill middle
    for i in range(1, m + 1):
        for j in range(1, k + 1):
            # The minimal error of j intervals on the first i points:

            # Exhaust all the options for the last interval. Each interval boundary is marked as either
            # 0 (Before first point), 1 (after first point, before second), ..., m (after last point)
            options = []
            for l in range(0, i + 1):
                next_errors = E[l, j - 1] + (cy[i] - cy[l]) + concatenate(
                    [[0], cumsum((-1) ** (ys[arange(l, i)] == 1))])
                min_error = argmin(next_errors)
                options.append((next_errors[min_error], (l, arange(l, i + 1)[min_error])))

            E[i, j], P[i][j] = min(options)

    # Extract best interval set and its error count
    best = []
    cur = P[m][k]
    for i in range(k, 0, -1):
        best.append(cur)
        cur = P[cur[0]][i - 1]
        if cur == None:
            break
    best = sorted(best)
    besterror = E[m, k]

    # Convert interval boundaries to numbers in [0,1]
    exs = concatenate([[0], xs, [1]])
    representatives = (exs[1:] + exs[:-1]) / 2.0
    intervals = [(representatives[l], representatives[u]) for l, u in best]

    return intervals, besterror


def main():
    # section a
    points = generatePoints(100)
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Section a: blue - 0.25,0,5-0,75 markers, red - intervals')
    plt.plot(points[0], points[1], 'ro')
    plt.axis([0, 1, -0.1, 1.1])
    plt.axvline(0.25, -0.1, 1.1, color='b')
    plt.axvline(0.5, -0.1, 1.1, color='b')
    plt.axvline(0.75, -0.1, 1.1, color='b')

    k = 2
    intervals, besterror = find_best_interval(points[0], points[1], k)

    plt.axvline(intervals[0][0], -0.1, 1.05, color='r')
    plt.axvline(intervals[0][1], -0.1, 1.05, color='r')
    plt.axvline(intervals[1][0], -0.1, 1.05, color='r')
    plt.axvline(intervals[1][1], -0.1, 1.05, color='r')
    plt.axhline(1.05, intervals[0][0], intervals[0][1], color='r')
    plt.axhline(1.05, intervals[1][0], intervals[1][1], color='r')
    plt.savefig('Q2_Section_A')

    # section c
    k = 2
    m = [i for i in range(10, 105, 5)]
    T = 100
    experiments = np.array([[0.1 for i in range(T)], [0.1 for i in range(T)]])  # true error, empirical error, m[i]
    plot_array_section_c = np.array([[0.1 for i in range(len(m))], [0.1 for i in range(len(m))]])
    for i in range(len(m)):
        for j in range(T):
            points = generatePoints(m[i])
            intervals, besterror = find_best_interval(points[0], points[1], k)
            experiments[0][j] = calcTrueError(intervals)
            experiments[1][j] = calcEmpiricalError(intervals, points)
        print("mean for: ", m[i], " true error: ", experiments[0][i], " empirical error: ", experiments[1][j])
        (plot_array_section_c[0][i]), plot_array_section_c[1][i] = np.mean(experiments, axis=1)

    plt.figure(2)
    plt.plot(m, plot_array_section_c[0], 'r', m, plot_array_section_c[1], 'b')
    plt.title('Section c: red - true error, blue - empirical error')
    plt.xlabel('samples number')
    plt.ylabel('error')
    plt.savefig('Q2_Section_C')

    # section D
    max_k = 20
    k_array = [i for i in range(1, max_k + 1)]
    error_array = [0.0 for i in range(max_k)]
    m = 50
    points = generatePoints(m)
    for i in range(max_k):
        intervals, besterror = find_best_interval(points[0], points[1], k_array[i])
        error_array[i] = calcEmpiricalError(intervals, points)

    plt.figure(3)
    plt.plot(k_array, error_array)
    plt.title('Section d')
    plt.xlabel('k')
    plt.ylabel('error')
    plt.savefig('Q2_Section_D')

    # section e
    T = 100
    experiments = np.array([[0.1 for i in range(T)], [0.1 for i in range(T)]])  # true error, empirical error, m[i]
    plot_array_section_e = np.array([[0.1 for i in range(1, max_k + 1)], [0.1 for i in range(1, max_k + 1)]])
    for i in range(max_k):
        for j in range(T):
            points = generatePoints(m)
            intervals, besterror = find_best_interval(points[0], points[1], k_array[i])
            experiments[0][j] = calcTrueError(intervals)
            experiments[1][j] = calcEmpiricalError(intervals, points)
        plot_array_section_e[0][i], plot_array_section_e[1][i] = np.mean(experiments, axis=1)
        print("mean for k=: ", k_array[i], " true error: ", plot_array_section_e[0][i], " empirical error: ",
              plot_array_section_e[1][i])

    plt.figure(4)
    plt.plot(k_array, plot_array_section_e[0], 'r', k_array, plot_array_section_e[1], 'b')
    plt.title('Section e: red - true error, blue - empirical error')
    plt.xlabel('k')
    plt.ylabel('error')
    plt.savefig('Q2_Section_E')

    # section f
    # choosing 10-fold cross validation (5 samples each) for determining k
    samples_per_fold = 5
    points = generatePoints(m)
    indexes = np.array([i for i in range(m)])
    random.shuffle(indexes)
    train = [np.array([0.0 for i in range(m - samples_per_fold)]), np.array([0 for i in range(m - samples_per_fold)])]
    test = [np.array([0.0 for i in range(samples_per_fold)]), np.array([0 for i in range(samples_per_fold)])]
    error_array = [0.0 for i in range(max_k)]
    for k in range(1, 21):
        comulative_error = 0.0
        for i in range(0, 50, samples_per_fold):
            train_index = 0
            test_index = 0
            for j in range(m):
                if j not in indexes[i:i + samples_per_fold]:
                    train[0][train_index], train[1][train_index] = points[0][j], points[1][j]
                    train_index += 1
                else:
                    test[0][test_index], test[1][test_index] = points[0][j], points[1][j]
                    test_index += 1
            intervals, besterror = find_best_interval(train[0], train[1], k)
            comulative_error += calcEmpiricalError(intervals, test)
        error_array[k - 1] = comulative_error * samples_per_fold / m
        print("finished ", samples_per_fold / m, " fold validation for k=", k, " with error: ",
              error_array[k - 1])

    plt.figure(5)
    plt.axis([0, 21, 0, error_array[19] + 0.2])
    plt.title('Section f: 10-fold cross validation')
    plt.plot(k_array, error_array)
    plt.xlabel('k')
    plt.ylabel('averaged error')
    plt.savefig('Q2_Section_F')


def generatePoints(m):
    x = np.random.uniform(0, 1, m)
    x = np.sort(x)
    n = 1
    y = np.array([np.random.binomial(n, 0.8) if (xi >= 0 and xi <= 0.25) or (
        xi >= 0.5 and xi <= 0.75) else np.random.binomial(n, 0.1) for xi in x])
    return [x, y]


# Section b
def calcTrueError(intervals):
    flatten_intervals = np.array([0.0], dtype='f')
    flatten_intervals = np.append(flatten_intervals, np.array([tup[i] for tup in intervals for i in range(2)]))
    flatten_intervals = np.append(flatten_intervals, 1.0)

    true_error = 0
    i = 0
    # calcultaing true errors. i % 2 == 1: right side of an interval. i % 2 == 0: left side
    # 0-0.25
    while (flatten_intervals[i] < 0.25):
        i += 1
        error_possibility = 0.8 if (i % 2 == 1) else 0.2
        true_error += error_possibility * (min(flatten_intervals[i], 0.25) - flatten_intervals[i - 1])

    # 0.25-0.5
    error_possibility = 0.1 if (i % 2 == 1) else 0.9
    true_error += error_possibility * (min(flatten_intervals[i], 0.5) - 0.25)

    while (flatten_intervals[i] < 0.5):  #
        i += 1
        error_possibility = 0.1 if (i % 2 == 1) else 0.9
        true_error += error_possibility * (min(flatten_intervals[i], 0.5) - flatten_intervals[i - 1])

    # 0.5-0.75
    error_possibility = 0.8 if (i % 2 == 1) else 0.2
    true_error += error_possibility * (min(flatten_intervals[i], 0.75) - 0.5)
    while (flatten_intervals[i] < 0.75):
        i += 1
        error_possibility = 0.8 if (i % 2 == 1) else 0.2
        true_error += error_possibility * (min(flatten_intervals[i], 0.75) - flatten_intervals[i - 1])

    # 0.75-1
    error_possibility = 0.1 if (i % 2 == 1) else 0.9
    true_error += error_possibility * (min(flatten_intervals[i], 1) - 0.75)
    while (flatten_intervals[i] < 1):  #
        i += 1
        error_possibility = 0.1 if (i % 2 == 1) else 0.9
        true_error += error_possibility * (min(flatten_intervals[i], 1) - flatten_intervals[i - 1])

    return true_error


def calcEmpiricalError(intervals, points):
    flatten_intervals = np.array([0.0], dtype='f')
    flatten_intervals = np.append(flatten_intervals, np.array([tup[i] for tup in intervals for i in range(2)]))
    flatten_intervals = np.append(flatten_intervals, 1.0)

    interval_index = 0
    sample_index = 0
    len_f = len(flatten_intervals)
    m = len(points[0])
    count = 0

    while (interval_index < len_f - 2 and sample_index < m):
        while (points[0][sample_index] > flatten_intervals[interval_index + 1]):
            interval_index += 1
        estimation = interval_index % 2
        if estimation != points[1][sample_index]:
            count += 1
        sample_index += 1

    return (float(count) / m)


if __name__ == '__main__':
    sys.exit(main())
