'''
Chaoyu Li
Date:4/8/2022
'''

import matplotlib.pyplot as plt
import numpy as np
import cvxopt
from matplotlib.colors import ListedColormap

def train_test_split(X, y, test_size, randomseed):
    np.random.seed(randomseed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    return X_train, X_test, y_train, y_test

def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy

def linear_kernel(**kwargs):
    def f(x1, x2):
        return np.inner(x1, x2)
    return f

def polynomial_kernel(power, coef, **kwargs):
    def f(x1, x2):
        return (np.inner(x1, x2) + coef)**power
    return f

class SVM(object):
    def __init__(self, C=None, kernel=linear_kernel, power=4, coef=1, isLinear=True):
        self.C = C # Penalty term(soft marginal).
        self.kernel = kernel # polynomial or linear.
        self.power = power # degree of kernel polynomial
        self.coef = coef # bias term in kernel polynomial
        self.alpha = None # alpha lagrange multiplier
        self.sv = None
        self.sv_label = None
        self.intercept = None
        self.isLinear = isLinear
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.weights = np.zeros((1, n_features))

        # Initialize kernel method with parameters
        self.kernel = self.kernel(power=self.power, coef=self.coef)

        # Calculate kernel matrix
        kernel_mat = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_mat[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem to solve for optimal alpha_i
        # minimize 1/2* xT * Q * x + pT * x subject to G*x <= h and A*x = b
        # here the problem is
        # maximize L = sum_over_i(a_i) - 1/2 * sum_over_i*sum_over_j(a_i*a_j*y_i*y_j*dot(xSV_i, xSV_j))
        # or Q = y_i * y_j * dot(xSV_i, xSV_j) and pT = -1
        Q = cvxopt.matrix(np.outer(y, y) * kernel_mat, tc='d')
        p = cvxopt.matrix(np.ones(n_samples) * -1)

        # subject to: sum_over_i(a_i *y_i) = 0 (from dL/db = 0)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        # subject to: -a_i <= 0 and a_i <= penalty_term
        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        cvx_opt_solution = cvxopt.solvers.qp(Q, p, G, h, A, b)

        # Lagrange multipliers alpha
        all_alpha = np.ravel(cvx_opt_solution['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = all_alpha > 1e-7

        # Get the corresponding lagr. multipliers
        self.alpha = all_alpha[idx]

        # Get the samples that will act as support vectors
        self.sv = X[idx]

        # Get the corresponding labels
        self.sv_label = y[idx]

        # Calculate intercept/bias term with first support vector
        #  bias term in SVM should NOT be regularized.
        self.intercept = self.sv_label[0]

        # apply complementary slackness:
        # b* = yi - sum_over_j (yj*alpha_j*k(xj, xi))
        for i in range(len(self.alpha)):
            self.intercept -= self.alpha[i] * self.sv_label[i] * \
                              self.kernel(self.sv[i], self.sv[0])

        # get weights: w* = sum_over_n (a_n*y_n*kernel(x_n))
        # for j in range(n_features):
        #     for i in range(len(self.alpha)):
        #         self.weights[j] += self.alpha[i] * self.sv_label[
        #                 i] * self.kernel(self.sv[i][j], self.sv[i][j])
        self.weights = np.sum(self.alpha.reshape((len(self.alpha),1))*
                              self.sv_label.reshape((len(self.alpha),1))*
                              self.sv, 0)[:, np.newaxis]

    # classify a point x via sign(b+w*x) = sign(b + sum_over_i(a_i*y_i*kernel(x_i,x))
    # with i loops through all support vector points
    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make preds
        for sample in X:
            pred = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.alpha)):

                pred += self.alpha[i] * self.sv_label[
                    i] * self.kernel(self.sv[i], sample)
            pred += self.intercept

            # make pred based on sign
            y_pred.append(np.sign(pred))
        return np.array(y_pred)
    
    """
    to display linear svm plane
    """
    def plot_linear_svm(self, X, Y):
        # plot the svm decision plane
        xx = np.linspace(0, 1, 11)
        a = -self.weights[0] / self.weights[1]
        yy = []
        for x in xx:
            yy.append(-(self.weights[0] * x + self.intercept) / self.weights[1])
        b = self.sv[0]
        yy_down = a * xx + (b[1] - a * b[0])
        b = self.sv[1]
        yy_up = a * xx + (b[1] - a * b[0])
        plt.plot(xx, yy)
        plt.plot(xx, yy_down, 'k--')
        plt.plot(xx, yy_up, 'k--')
        plt.savefig('linearSVM.png')
        plt.close()

    def plot_nonlinear_svm(self):
        h = 0.1
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        grid = np.array([xx.ravel(), yy.ravel()]).T
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.1, cmap = ListedColormap(('yellow', 'green')))
        plt.savefig('nonlinearSVM.png')
        plt.close()
    
    """
    display plots
    """
    def display(self, X, y):
        print("W:",self.weights)
        print("Support_vectors:")
        print(self.sv)
        print(self.sv_label)

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=1, s=10, edgecolors='k')
        plt.scatter(self.sv[:, 0], self.sv[:, 1], facecolors='none', s=100, edgecolors='k')

        if (self.isLinear):
            self.plot_linear_svm(X, y)
        else:
            self.plot_nonlinear_svm()

if __name__ == "__main__":
    # import data
    data_linear = np.loadtxt('linsep.txt', dtype='float', delimiter=',')
    data_nonlinear = np.loadtxt('nonlinsep.txt', dtype='float', delimiter=',')

    # Linear SVM
    print("Linear:")
    # prepare linear training and test data
    X = data_linear[:, 0:2]
    y = data_linear[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, randomseed=7)

    # plot original linear data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=1, s=20, edgecolors='k')
    plt.savefig('original_linear.png')
    plt.close()

    # SVM model
    svm = SVM(kernel=linear_kernel, isLinear=True)

    # train and predict svm
    svm.fit(X_train, y_train)
    svm.display(X_train, y_train)
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    # calculate accuracy
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print("Training Accuracy:", accuracy_train)
    print("Testing Accuracy:", accuracy_test)

    # Non-Linear SVM
    print("Non-Linear:")
    # prepare non-linear training and test data
    X = data_nonlinear[:, 0:2]
    y = data_nonlinear[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, randomseed=7)

    # plot original linear data
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=1, s=20, edgecolors='k')
    plt.savefig('original_nonlinear.png')
    plt.close()

    # SVM model
    svm = SVM(kernel=polynomial_kernel, power=2, coef=0, isLinear=False)

    # train and predict svm
    svm.fit(X_train, y_train)
    svm.display(X_train, y_train)
    y_train_pred = svm.predict(X_train)
    y_test_pred = svm.predict(X_test)

    # calculate accuracy
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    print("Training Accuracy:", accuracy_train)
    print("Testing Accuracy:", accuracy_test)