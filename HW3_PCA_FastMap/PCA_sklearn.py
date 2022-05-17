'''
Chaoyu Li
Date: 2/18/2022
'''
from numpy import genfromtxt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dimension = 2
if __name__ == '__main__':
    X = genfromtxt('pca-data.txt', delimiter='\t')
    pca = PCA(n_components=dimension)
    pca.fit(X)
    for idx, pc in enumerate(pca.components_):
        print('Vector {0}: {1}'.format(idx + 1, pc))
    transformed = pca.transform(X)
    print("result:\n", transformed)
    fig, ax = plt.subplots()
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    projection = ax.scatter(transformed[:, 0], transformed[:, 1], c='g')
    ax.legend([projection], ['Projected data'])
    plt.savefig('2dData_sklearn.png')
    plt.close()