#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:46:38 2020

@author: youpele
"""

# import pandas as pd
# import hdbscan
from dtaidistance import dtw
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import inf
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score



class Clustering:
    
    def get_dtw(df, plot_path):
        '''
        Returns the distance matrix of a given dataframe.
        
        df : Path to the dataframe.
        plot_path: Path to where the plots would be saved. 
        
        '''
        
        df_1 = df.to_numpy()
        df_1 = np.transpose(df_1)
        
        # Distance matrics
        print("Computing distance matrix...")
        distance_matrix1= dtw.distance_matrix_fast(df_1)
        distance_matrix1 =  np.where(distance_matrix1==inf, 0, distance_matrix1) 
        print("Successfully computed the distance matrix.")
        
        # Plotting the distance_matrix1 
        sns.heatmap(distance_matrix1, cmap='Reds')
        plt.title("Distance matrix of the data")
        plt.savefig(fname = plot_path + "/dtw_distance_matrix.png", dpi = 300)
        
        # Plotting dendogram
        fig = plt.figure(figsize=(10, 7))
        plt.title("Data Dendograms")
        dend = shc.dendrogram(shc.linkage(df_1, method='ward'))
        fig.savefig(fname = plot_path + "/data_dendogram.png", dpi = 300)
        plt.close(fig)
        
        return distance_matrix1
    
    
    def get_clusters_ari(df_1, df_2, n_clusters = 4):
        '''
        Returns the adjusted rand score of two dataframes.
        
        df_1: The first dataframe.
        df_2: The second dataframe.
        n_clusters: The number of clusters to be used for both dataframes.
        '''
        
        
        df_1 = df_1.to_numpy()
        df_1 = np.transpose(df_1)
        
        # Processing the first dataframe
        cluster1 = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster1.fit_predict(df_1)

        #UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(df_1)
        embedding.shape
        
                
        # Viewing the plots alongside the umapped dataframe
        plt.scatter(embedding[:, 0], embedding[:, 1], c=cluster1.labels_, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
        plt.title('UMAP projection of the first dataframe', fontsize=12);
        
        # Processing the second dataframe
        cluster2 = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        cluster2.fit_predict(df_2)

        # Ajusted rand score
        ars = adjusted_rand_score(cluster1.labels_, cluster2.labels_ )
        
        return ars


    


        
        

        
        

        
