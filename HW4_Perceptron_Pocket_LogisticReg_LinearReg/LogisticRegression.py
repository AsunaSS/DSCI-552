'''
Chaoyu Li
Date: 3/4/2022
'''
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01
numIteration = 7000
filename = "classification.txt"
np.random.seed(8)

def calcSigmoid(z):
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid

def calcGradient(weights, x, y):
    grad = (x * y) * calcSigmoid(-y * np.dot(weights, x))
    return grad

def train(X, Y_train, lr, maxIteration):
    dimension = X.shape[1]
    num = X.shape[0]
    error_list = []
    X = np.insert(X, 0, 1, axis=1)
    # Initialize weights randomly
    weights = np.random.random(dimension + 1)

    iteration = 0
    while iteration < maxIteration:
        grad = np.zeros(dimension + 1)
        for x, y in zip(X, Y_train):
            descent = calcGradient(weights, x, y)
            grad = np.add(grad, descent)

        grad = grad / num
        weights += lr * grad
        temp = np.sign(np.dot(X, weights))
        error_list.append(num - np.where(temp == Y_train)[0].shape[0])

        if iteration % 500 == 0:
            print('Iteration in Progress: {0}'.format(iteration))
            print('Error Number: {0}'.format(error_list[-1]))
        iteration += 1
        # If no more misclassification, break
        if error_list[-1] == 0:
            break

    return iteration, weights, error_list


def predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)
    y_predicted = [None for i in range(X.shape[0])]
    for index, x in enumerate(X):
        prob = calcSigmoid(1 * np.dot(weights, x))
        if prob <= 0.5:
            y_predicted[index] = -1
        else:
            y_predicted[index] = 1
    return np.asarray(y_predicted)


# Only use col 0,1,2,4
def loadTXT(filename):
    trainData = np.loadtxt(filename, delimiter=',', dtype='float', usecols=(0, 1, 2, 4))
    return trainData


# Plot Error Rate
def plotError(error_list):
    plt.ylabel('number of error points')
    plt.xlabel('iterations')
    plt.plot(error_list, color='red', linestyle='dashed', linewidth=2, markersize=2)
    plt.savefig('LogisticError.png')
    plt.close()


# Plot 3D classification result
def plot3D(X, y_train, weights):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    for i in range(len(X)):
        _x = X[i][0]
        _y = X[i][1]
        _z = X[i][2]
        if y_train[i] == 1:
            _c_p = ax.scatter(_x, _y, _z, c='c', marker='^')
        else:
            _r_p = ax.scatter(_x, _y, _z, c='r', marker='o')
    ax2 = plt.gca()
    ax.legend([_c_p, _r_p], ['Label 1 data', 'Label -1 data'])
    # Plot the classification surface
    x, y = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
    z = (-weights[0] - weights[1] * x - weights[2] * y) / weights[3]
    surface = ax2.plot_surface(x, y, z, color='b', alpha=0.3)
    plt.savefig('Logistic3DPlot.png')
    plt.close()


if __name__ == '__main__':
    # Initiation
    trainData = loadTXT(filename)
    X = trainData[:, :-1]
    y_train = trainData[:, -1]

    # Logistic Regression Algorithm
    numIteration, finalWeights, error_list = train(X, y_train, lr, numIteration)
    y_predicted = predict(X, finalWeights)
    accuracy = np.where(y_predicted == y_train)[0].shape[0] / y_predicted.shape[0]

    # Print result
    print('Number of Iterations: {0}'.format(numIteration))
    print('Accuracy: {0}'.format(accuracy))
    print('Final Weights [W0, W1, W2, W3]: {0}'.format(finalWeights))

    plotError(error_list)
    plot3D(X, y_train, finalWeights)