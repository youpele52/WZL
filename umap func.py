#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:14:46 2020

@author: youpele
"""

import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np


digits = load_digits()

embedding = umap.UMAP(n_neighbors=15,
                      metric='correlation', 
                      n_components = 2).fit_transform(digits.data)



#plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))

plt.scatter(embedding[:, 0], embedding[:, 1],)
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Iris dataset');






def umap_projection (df, data_title = "", n_components = 2):
    
    '''
    
    This function reduces a multi feature datasets to a datasets with n_components. 
    It accepts a dataframe, data title and n_components.
    '''
    
    
    embedding = umap.UMAP(n_neighbors=15,
                      metric='correlation', 
                      n_components = n_components).fit_transform(df)
    
    if n_components == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1],)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the {a} data.'.format(a = data_title))
        
    return embedding


go = umap_projection(df = load_digits().data, n_components=4)


g_2 = umap_projection(df = go, n_components=2)

        
        
        

    
    
    