'''
Chaoyu Li
Date: 3/5/2022
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def getInputData(filename):
    _data = np.genfromtxt(filename, delimiter=',')
    _X = _data[:, :2]
    _Z = _data[:, 2]
    return _X, _Z

def plot3D(X, Z, lr):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    x = X[:, 0]
    y = X[:, 1]
    z = np.array(Z[:])

    ax.scatter(x, y, z, c='r', marker='o')

    xx, yy = np.meshgrid(np.arange(x.min()-0.25, x.max()+0.25, 0.02), np.arange(y.min()-0.25, y.max()+0.25, 0.02))
    zz = np.zeros(shape=(xx.shape))
    for i in range(len(xx)):
        for j in range(len(xx[i])):
            zz[i][j] = lr.predict([[xx[i][j], yy[i][j]]])
    ax.plot_surface(xx, yy, zz, color='b', alpha=0.3)
    plt.savefig('Linear3DPlot_sklearn.png')
    plt.close()


if __name__ == '__main__':
    X, Z = getInputData("linear-regression.txt")
    lr = LinearRegression()
    lr.fit(X, Z)
    #print(str(lr.get_params))
    #print(lr.score(X,Z))
    print('Linear Regression Weights: {0}'.format(lr.coef_))
    plot3D(X, Z, lr)
