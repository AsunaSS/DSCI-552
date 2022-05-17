'''
Chaoyu Li
Date: 3/5/2022
'''
import numpy as np
import matplotlib.pyplot as plt

filename = "linear-regression.txt"

def train(X, Y_train):
    weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), Y_train)
    return weights

def loadTXT(filename):
    trainData = np.loadtxt(filename, delimiter=',', dtype='float', usecols=(0, 1, 2))
    return trainData

# Plot 3D classification result
def plot3D(X, y_train, weights):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    x = X[:, 0]
    y = X[:, 1]
    z = np.array(y_train[:])
    ax.scatter(x, y, z, c='r', marker='o')
    ax2 = plt.gca()
    # Plot the classification surface
    xx, yy = np.meshgrid(np.arange(x.min()-0.25, x.max()+0.25, 0.02), np.arange(y.min()-0.25, y.max()+0.25, 0.02))
    zz = weights[0] * xx + weights[1] * yy
    surface = ax2.plot_surface(xx, yy, zz, color='b', alpha=0.3)
    plt.savefig('Linear3DPlot.png')
    plt.close()


if __name__ == '__main__':
    # Initiation
    trainData = loadTXT(filename)
    X = trainData[:, :-1]
    y_train = trainData[:, -1]

    # Linear Regression Algorithm
    finalWeights = train(X, y_train)

    # Print result
    print('Final Weights [W0, W1]: {0}'.format(finalWeights))

    plot3D(X, y_train, finalWeights)