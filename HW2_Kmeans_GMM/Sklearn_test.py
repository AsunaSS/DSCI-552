'''
Chaoyu Li
Date: 2/9/2022
'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import mixture

trials = 10 #Trail numbers for each algorithm.
clusters = 3 #Cluster numbers

X = np.genfromtxt('clusters.txt', delimiter=',')

km = KMeans(n_clusters=clusters, n_init=trials, algorithm ='full')
km.fit(X)

centroids = km.cluster_centers_
labels = km.labels_

# Max 10 clusters can be marked by 5 different colors
colors = ["g.", "b.", "r."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=12)

plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=20, linewidths=10)
plt.savefig("Sklearn_Kmeans_cluster.png")

print ('centroids=', centroids, '\n\n\n')

#GMM
plt.figure()
gmm = mixture.GaussianMixture(n_components=clusters, n_init=trials, covariance_type="full")
gmm.fit(X)
labels = gmm.predict(X)
weights = gmm.weights_
means = gmm.means_
n_cov = gmm.covariances_

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=12)

plt.scatter(means[:,0], means[:,1], marker='x', s=20, linewidths=10)
plt.savefig("Sklearn_GMM_cluster.png")

print ('GMM weights:', weights)
print ('GMM means:', means)
print ('GMM covars: components=', n_cov)