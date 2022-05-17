'''
Chaoyu Li
Date: 2/9/2022
'''
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import style

k = 3 #k of k-means
path = "clusters.txt"

def Kmeans(k, dimension, means_list, n):
    #initialize
    clusterNumber = np.zeros((k, dimension))
    clusterSize = np.zeros(k)
    meansNumber = np.zeros((k, dimension))
    clusterBelong = n * [None]
    cluster_dict = {}
    cluster_dict["0"] = []
    cluster_dict["1"] = []
    cluster_dict["2"] = []
    iteration = 0

    #algorithm
    for i in range(k):
        meansNumber[i] = points[means_list[i]]
    #If iteration > 10000 but still not get stable, break
    while iteration < 10000:
        cluster_dict = {}
        cluster_dict["0"] = []
        cluster_dict["1"] = []
        cluster_dict["2"] = []
        changenode = 0
        clusterNumber.fill(0)
        clusterSize.fill(0)
        for i in range(n):
            clusterDistance = np.array([])
            for means in meansNumber:
                clusterDistance = np.append(clusterDistance, (np.sum((np.array(points[i]) - np.array(means)) ** 2)))
            index = np.argmin(clusterDistance)
            cluster_dict[str(index)].append(i)
            clusterNumber[index] += points[i]
            clusterSize[index] += 1
            if (clusterBelong[i] == None or index != clusterBelong[i]):
                changenode += 1
                clusterBelong[i] = index
        for i in range(k):
            meansNumber[i] = np.divide(clusterNumber[i], clusterSize[i])
        #If NOTHING change in one iteration, get the result
        if changenode == 0:
            #print(clusterBelong)
            return meansNumber, cluster_dict, clusterSize, clusterBelong
        iteration += 1
    return meansNumber, cluster_dict, clusterSize, clusterBelong

#Value for xi in normal distribution 'k'
def normal(xI, muK, sigmaK, D):
    p = 1 / pow((2 * math.pi), -D/2) * pow(abs(np.linalg.det(sigmaK)), -1/2) * np.exp(-1/2 * np.dot(np.dot((xI - muK).T, np.linalg.inv(sigmaK)), (xI - muK)))
    return p

#Get maximum like
def maximizeL(pointNumber, k, alpha, points, mu, sigma):
    new_like = 0
    clusterBelong = np.zeros(k)
    for i in range(pointNumber):
        clusterBelong = np.zeros(k)
        sum = 0
        for j in range(k):
            temp = alpha[j] * normal(points[i].T, mu[j].T, sigma[j], points.shape[1])
            clusterBelong[j] = temp
            #sum += alpha[j] * normal(points[i].T, mu[j].T, sigma[j], points.shape[1])
        sum = np.sum(clusterBelong)
        clusterBelong = np.divide(clusterBelong, sum)
        if clustersBelong[i] != np.argmax(clusterBelong):
            clustersBelong[i] = np.argmax(clusterBelong)
        new_like += np.log(sum)
    #print("New_like:", new_like)
    return new_like

def Estep(pointNumber, k, W, alpha, points, mu, sigma):
    sum = np.zeros(pointNumber)
    for i in range(pointNumber):
        temp = np.zeros(k)
        for j in range(k):
            temp[j] = float(alpha[j]) * normal(points[i].T, mu[j].T, sigma[j], points.shape[1])
            sum[i] += temp[j]
        for j in range(k):
            W[j][i] = temp[j] / sum[i]

def Mstep(pointNumber, k, W, alpha, points, mu, sigma):
    for K in range(k):
        alpha[K] = np.sum(W[K]) / pointNumber

        total = np.zeros(mu.shape[1])
        for i in range(pointNumber):
            total += W[K][i] * points[i]
        mu[K] = total / np.sum(W[K])
        #print(W)
        Sum = np.zeros([points.shape[1], points.shape[1]])
        for i in range(pointNumber):
            if points[i].ndim == 1:
                #Reshape the 1-dimension array to a 2*1 matrix
                diff = points[i].reshape(points.shape[1], 1) - mu[K].reshape(mu.shape[1], 1)
                Sum += W[K][i] * np.dot(diff, diff.T)
            else:
                Sum += W[K][i] * np.dot(points[i] - mu[i], (points[i] - mu[i]).T)
        sigma[K] = Sum / np.sum(W[K])

def GMM(k, points, alpha, mu, sigma):
    points = np.array(points)
    pointNumber = len(points)
    W = np.zeros([k, pointNumber]) #weight of point
    like = None
    new_like = maximizeL(pointNumber, k, alpha, points, mu, sigma)
    iteration = 0
    while ((iteration == 0) or (new_like - like > 5e-4)):
        like = new_like
        Estep(pointNumber, k, W, alpha, points, mu, sigma)
        Mstep(pointNumber, k, W, alpha, points, mu, sigma)
        new_like = maximizeL(pointNumber, k, alpha, points, mu, sigma)
        iteration += 1
    print("Total Recursion: ", iteration)
    return like, alpha, mu, sigma

def loadTXT(filename):
    file = open(filename, "r")
    points=[]
    for line in file:
        points.append([float(x) for x in line.split(',')])
    return points

if __name__ == '__main__':
    points = loadTXT(path)
    #print(points)
    n = len(points)
    dimension = len(points[0])
    #initial mean
    means = []
    means.append(random.randint(0, n - 1))
    while len(means) < k:
        temp = random.randint(0, n - 1)
        if temp not in means:
            means.append(temp)

    #K-means
    result, clusters, clustersSize, clustersBelong = Kmeans(k, dimension, means, n)
    #print(clusters)
    print(clustersSize)
    print("K-Means centroid is\n", result)
    #print()
    #print(clustersBelong)
    print()
    fig, ax = plt.subplots()
    types = []
    colors = ['g','b','r']
    #print(colors)
    for i, color in enumerate(colors):
        #print(i)
        #print(clusters[str(i)])
        for index in clusters[str(i)]:
            #print(points[index][0], points[index][1])
            ax.scatter(points[index][0], points[index][1], c=color)
    for centroid in result:
        ax.scatter(centroid[0], centroid[1], c='k', marker='^', linewidths=8, norm=0.3)
    plt.savefig("Kmeans_cluster.png")

    #GMM
    Alphas = clustersSize / 150 #Amplitudes
    Sigmas = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]) #Covar
    Mus = result #Means
    like, alpha, mu, sigma = GMM(k, points, Alphas, Mus, Sigmas)
    #print(clustersBelong)
    clusters_dict = {}
    clusters_dict["0"] = []
    clusters_dict["1"] = []
    clusters_dict["2"] = []
    for i in range(len(clustersBelong)):
        clusters_dict[str(clustersBelong[i])].append(i)
    #print(clusters_dict)
    print("The amplitudes are:", alpha)
    print("The means are:", mu)
    print("The covariances are:", sigma)

    fig, ax = plt.subplots()
    types = []
    colors = ['g', 'b', 'r']
    # print(colors)
    for i, color in enumerate(colors):
        # print(i)
        # print(clusters[str(i)])
        for index in clusters_dict[str(i)]:
            # print(points[index][0], points[index][1])
            ax.scatter(points[index][0], points[index][1], c=color)
    for centroid in mu:
        ax.scatter(centroid[0], centroid[1], c='k', marker='^', linewidths=8, norm=0.3)
    plt.savefig("GMM_cluster.png")


'''
[31. 33. 86.]
K-Means centroid is
 [[ 5.73849535  5.16483808]
 [ 3.28884856  1.93268837]
 [-0.96065291 -0.65221841]]

Total Recursion:  18
The amplitudes are: [0.22596436 0.2080321  0.56600354]
The means are: [[ 5.52794237  4.89659356]
 [ 3.1917278   1.8040412 ]
 [-0.97944925 -0.64151781]]
The covariances are: [[[ 2.31349825  0.28509797]
  [ 0.28509797  2.21584433]]

 [[ 2.02585542  0.18780605]
  [ 0.18780605  2.61775025]]

 [[ 1.20316601 -0.09844353]
  [-0.09844353  2.01675539]]]'''