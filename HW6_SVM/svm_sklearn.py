'''
Chaoyu Li
Date:4/8/2022
'''

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def train_test_split(X, y, test_size, seed=None):
    np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_linear = np.loadtxt('linsep.txt', dtype='float', delimiter=',')
    data_nonlinear = np.loadtxt('nonlinsep.txt', dtype='float', delimiter=',')
    print("Linear:")
    X = data_linear[:, 0:2]
    y = data_linear[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, seed=7)
    clf_linear = svm.SVC(C=10.0, kernel='linear', degree=3, gamma='scale',
                         coef0=0.0, shrinking=True, probability=False, tol=0.00001,
                         cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                         decision_function_shape='ovr', random_state=None)

    clf_linear.fit(X_train, y_train)

    support_vectors = clf_linear.support_vectors_
    support_vector_indices = clf_linear.support_

    print("support vectors: ", support_vectors)
    print("weights: ", clf_linear.coef_)
    y_test_pred = clf_linear.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print("Testing Accuracy:", accuracy_test)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=1, s=10)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=100, edgecolors='k')

    inputX = np.linspace(0, 1, 11)
    outputY = []
    for x in inputX:
        outputY.append(-(clf_linear.coef_[0][0] * x + clf_linear.intercept_[0]) / clf_linear.coef_[0][1])
    plt.plot(inputX, outputY)
    plt.savefig('linearSVM_sklearn.png')
    plt.close()

    print("Non-Linear:")
    X = data_nonlinear[:, 0:2]
    y = data_nonlinear[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, seed=7)
    clf_nonlinear = svm.SVC(C=1000.0, kernel='poly', degree=2, gamma='scale',
                            coef0=0.0, shrinking=True, probability=False, tol=0.00001,
                            cache_size=200, class_weight=None, verbose=False, max_iter=-1,
                            decision_function_shape='ovr', random_state=None)
    clf_nonlinear.fit(X_train, y_train)
    support_vectors = clf_nonlinear.support_vectors_
    print("support vectors: ", support_vectors)
    y_test_pred = clf_nonlinear.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print("Testing Accuracy:", accuracy_test)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', alpha=1, s=10)
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', s=100, edgecolors='k')

    h = 0.1
    x_min, x_max = -10, 10
    y_min, y_max = -10, 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    inputX = np.array([xx.ravel(), yy.ravel()]).T
    Z = clf_nonlinear.predict(inputX)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.1)
    plt.savefig('nonlinearSVM_sklearn.png')
    plt.close()