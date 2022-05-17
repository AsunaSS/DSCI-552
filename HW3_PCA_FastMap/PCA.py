'''
Chaoyu Li
Date: 2/18/2022
'''
import numpy as np
import matplotlib.pyplot as plt

dimension = 2 # Here we need 2D data

# Calculate mean for each point and return each (point - self.mean)
def meanNormalization(points):
    Mean = np.mean(points, axis=0)
    return np.array(points - Mean)

# Calculate covariance matrix
def covarMatrix(points):
    Points = np.array(points)
    covar = np.cov(Points.T)
    return covar

# Sort eigenvalue by decreasing then return first k eigenvector, here k=2
def firstKEigenvector(covar, k):
    original_eigenvector, original_eigenvalue, vector = np.linalg.svd(covar)
    #print(vector)

    # Get eigenvalue index by decreasing order
    sorted_index = np.argsort(- original_eigenvalue)
    #print(sorted_index)
    
    # Get eigenvector & eigenvalue by decreasing order
    sorted_eigenvector = original_eigenvector[:, sorted_index]
    sorted_eigenvalue = original_eigenvalue[sorted_index]
    #print(sorted_eigenvector)
    #print("!@#!@#!@#")
    #print(sorted_eigenvalue)
    
    result = sorted_eigenvector[:, :k]
    return result

def PCA(points, dimensions):
    normal_points = meanNormalization(points)
    covar = covarMatrix(normal_points)
    vector_k = firstKEigenvector(covar, dimensions)

    # Project new points to k dimensions
    result = np.dot(vector_k.T, normal_points.T)
    return result.T, vector_k

def plotOriginal3D(points):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    original = ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.legend([original], ['Original data'])
    plt.savefig('3dData.png')
    plt.close()

def plotPCA2D(transPoints):
    fig, ax = plt.subplots()
    plt.xlabel('X Label')
    plt.ylabel('Y Label')
    projection = ax.scatter(transPoints[:, 0], transPoints[:, 1])
    ax.legend([projection], ['Projected data'])
    plt.savefig('2dData.png')
    plt.close()

def loadTXT(filename):
    points = np.genfromtxt(filename, delimiter='\t')
    return points


if __name__ == '__main__':
    # Read data
    points = loadTXT('pca-data.txt')
    #print(type(points))
    #print(points)
    
    # Plot original data in 3D space
    plotOriginal3D(points)

    # PCA
    result, vector_k = PCA(points, dimension)

    # Print results
    print('Sorted eigenvector #1= ', vector_k.T[0])
    print('Sorted eigenvector #2= ', vector_k.T[1])
    print('results=\n', result)

    # Plot projected data in 2D space
    plotPCA2D(result)