'''
Chaoyu Li
Date: 3/5/2022
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


lr = 0.01
numIteration = 7000
filename = "classification.txt"
np.random.seed(8)

if __name__ == '__main__':
    trainData = np.loadtxt(filename, delimiter=',', dtype='float', usecols=(0, 1, 2, 3, 4))
    X = trainData[:, :-1]
    y_train = trainData[:, -1]
    model = LogisticRegression()
    model.max_iter = numIteration
    model.fit(X, y_train)
    y_predicted = model.predict(X)
    print('Number of Iterations: {0}'.format(model.n_iter_))
    print('Best Accuracy: {0}'.format(model.score(X,y_train)))
    print('Logistic Regression Weights: {0}'.format(model.coef_))