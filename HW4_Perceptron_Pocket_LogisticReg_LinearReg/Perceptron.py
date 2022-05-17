'''
Chaoyu Li
Date: 3/4/2022
'''
import numpy as np
import matplotlib.pyplot as plt

lr = 0.01
filename = "classification.txt"
np.random.seed(8)

def train(X, Y_train, lr):
    dimension = X.shape[1]
    num = X.shape[0]
    error_list = []
    X = np.insert(X, 0, 1, axis=1)
    # Initialize weights randomly
    weights = np.random.random(dimension + 1)
    
    iteration = 0
    while True:
        for x, y in zip(X, Y_train):
            prod = np.dot(x, weights)
            if prod > 0 and y < 0:
                weights -= lr * x
            elif prod < 0 and y > 0:
                weights += lr * x
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
    return np.sign(np.dot(X, weights))

# Only use first four col
def loadTXT(filename):
    trainData = np.loadtxt(filename, delimiter=',', dtype='float', usecols=(0, 1, 2, 3))
    return trainData

# Plot Error Rate
def plotError(error_list):
    plt.ylabel('number of error points')
    plt.xlabel('iterations')
    plt.plot(error_list, color='red', linestyle='dashed', linewidth=2, markersize=2)
    plt.savefig('perceptronError.png')
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
    x, y = np.meshgrid(np.arange(-0.2, 1.2, 0.02), np.arange(-0.2, 1.2, 0.02))
    z = (-weights[0] - weights[1] * x - weights[2] * y) / weights[3]
    surface = ax2.plot_surface(x, y, z, color='b', alpha=0.3)
    plt.savefig('perceptron3DPlot.png')
    plt.close()

if __name__ == '__main__':
    # Initiation
    trainData = loadTXT(filename)
    X = trainData[:, :-1]
    y_train = trainData[:, -1]

    # Perceptron Algorithm
    numIteration, finalWeights, error_list = train(X, y_train, lr)
    y_predicted = predict(X, finalWeights)
    accuracy = np.where(y_predicted == y_train)[0].shape[0] / y_predicted.shape[0]

    # Print result
    print('Number of Iterations: {0}'.format(numIteration - 1))
    print('Accuracy: {0}'.format(accuracy))
    print('Final Weights [W0, W1, W2, W3]: {0}'.format(finalWeights))

    plotError(error_list)
    plot3D(X, y_train, finalWeights)