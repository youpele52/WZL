#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 12:04:09 2020

@author: youpele
"""




# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dico = pd.read_json("dico_features.json")
dico_2 = dico.drop(['way', 'index', 'segment'], axis=1)
X = dico_2

# this  replaces the NaN data point with 0
X = pd.DataFrame(X).fillna(value = 0, )

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()




# Using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch # importing the Hierarichical clustering lib
dendogram  = sch.dendrogram(sch.linkage(X, method = 'ward')) 




# Fitting Hierarichical clustering to the dataset, there are two classes to use the most common being AgglomerativeClustering and the other is Divisive
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 3, affinity= 'euclidean', linkage = 'ward')
y_hc= hc.fit_predict (X)

# NOTE IF YOU ARE WORKING WITH MULTI DIMENSIONAL DATA, YOU FIRST NEED TO REDUCE THE DIMENSION BEFORE VISUALIZING THE DATA
# Visualing the clusters 
plt.scatter(X[y_hc==0,0], X[y_hc==0,1], s =100, c= 'red', label = 'Cluster 1')  # careful client # the coordinate of all observation points that belongs to cluster one. The first X is the x coord, and the second is the y coord. s is the size, c is color
plt.scatter(X[y_hc==1,0], X[y_hc==1,1], s =100, c= 'green', label = 'Cluster 2') # standard spenders # second cluster
plt.scatter(X[y_hc==2,0], X[y_hc==2,1], s =100, c= 'blue', label = 'Cluster 3') # Target, because they have high income and spend more, thus more ads to them 
plt.scatter(X[y_hc==3,0], X[y_hc==3,1], s =100, c= 'cyan', label = 'Cluster 4') # careless
plt.scatter(X[y_hc==4,0], X[y_hc==4,1], s =100, c= 'pink', label = 'Cluster 5') # # sensible







# The Mutual Information is a measure of the similarity between two labels of the same data. 


import sklearn
sklearn.metrics.mutual_info_score(X[0],X[1])

from sklearn.metrics.cluster import normalized_mutual_info_score


normalized_mutual_info_score(X[0],X[1], average_method ='geometric' )

adjusted_mutual_info_score(X[0],X[1])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X = pca.fit_transform(X)


