'''
Chaoyu Li
Date: 3/5/2022
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

def getInputData(filename):
    _data = np.genfromtxt(filename, delimiter=',')
    _X = _data[:, :3]
    _Y = _data[:, 3]
    return _X, _Y

def getInputData1(filename):
    _data = np.genfromtxt(filename, delimiter=',')
    _X = _data[:, :3]
    _Y = _data[:, 4]
    return _X, _Y

def plot(X, Y, W):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    _data_set = X
    for i in range(len(_data_set)):
        _x = _data_set[i][1]
        _y = _data_set[i][2]
        _z = _data_set[i][3]
        if Y[i] == 1:
            _c_p = ax.scatter(_x, _y, _z, c='c', marker='^')
        else:
            _r_p = ax.scatter(_x, _y, _z, c='r', marker='o')

    ax.legend([_c_p, _r_p], ['Label 1 data', 'Label -1 data'])
    xx, yy = np.meshgrid(np.arange(-0.2, 1.2, 0.02), np.arange(-0.2, 1.2, 0.02))
    zz = -(W[0] + W[1] * xx + W[2] * yy) / W[3]
    ax.plot_surface(xx, yy, zz, color='b', alpha=0.3)  # Plot the segmentation plane
    plt.savefig('Perceptron3DPlot_sklearn.png')
    plt.close()


def plot_error(iterationTimes, errorNumber):
    _x = iterationTimes
    _y = errorNumber
    plt.plot(_x, _y)
    plt.xlabel('iteration')
    plt.ylabel('Error number')
    plt.savefig('PocketError_sklearn.png')
    plt.close()


if __name__ == '__main__':
    # Perceptron
    X, Y = getInputData("classification.txt")
    X = np.c_[np.ones(len(X)), np.array(X)]
    pla = Perceptron()
    pla.n_iter = 1
    pla.max_iter = 40000
    pla.alpha = 0.01
    info = pla.get_params()
    print(info)
    pla = pla.fit(X, Y)
    score = pla.score(X, Y)
    W = pla.coef_
    print('Accuracy: {0}'.format(score))
    print('Final Weights [W0, W1, W2, W3]: {0}'.format(W))
    plot(X, Y, W[0])

    # Pocket
    iterList = []
    numList = []
    best_score = 0
    W = None
    X, Y = getInputData1("classification.txt")  # Get column 5 as Y
    X = np.c_[np.ones(len(X)), np.array(X)]  # Coordinates vector of points, which is added '1' at first column.
    pla = Perceptron()
    pla.n_iter = 1
    pla.warm_start = True
    info = pla.get_params()
    for i in range(0, 7000):
        pla = pla.fit(X, Y)
        score = pla.score(X, Y)
        ErrorNum = (1 - score) * 2000
        iterList.append(i)
        numList.append(ErrorNum)
        if (best_score <= score or i == 0):
            best_score = score
            W = pla.coef_
    print('Accuracy: {0}'.format(best_score))
    print('Final Weights [W0, W1, W2, W3]: {0}'.format(W))
    plot_error(iterList, numList)