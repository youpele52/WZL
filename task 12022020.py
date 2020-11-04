#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:08:35 2020

@author: youpele
"""

#imports

import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



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
    
    return pd.DataFrame(embedding)



# Importing the dataset
        
dico = pd.read_json("dico_features.json")


# Punch Data 

Gegenhalter = dico.iloc[:,28:35]

Niederhalter = dico.iloc[:,35:56]
Niederhalter_1 = Niederhalter.iloc[:,0:7]
Niederhalter_2 = Niederhalter.iloc[:,7:14]
Niederhalter_4 = Niederhalter.iloc[:,14:22]

Stempel = dico.iloc[:,70:98]
Stempel_1 = Stempel.iloc[:, 0:7]
Stempel_2 = Stempel.iloc[:, 7:14]
Stempel_3 = Stempel.iloc[:, 14:21]
Stempel_4 = Stempel.iloc[:, 21:28]


Ge_Nie_Stem = pd.concat([Gegenhalter, Niederhalter, Stempel], axis=1)
Ge_Nie_Stem_umap = umap_projection(df = Ge_Nie_Stem, 
                                   data_title= 'Gegenhalter + Niederhalter + Stempel',n_components=4)

Ge_Nie_Stem_umap_2 = umap_projection(df = Ge_Nie_Stem_umap, 
                                   data_title= 'Gegenhalter + Niederhalter + Stempel',n_components=2)



Ge_Nie_Stem_coild_data = pd.concat([Ge_Nie_Stem_umap, coil_data], axis =1)
corrMatrix = Ge_Nie_Stem_coild_data.corr()

plt.matshow(corrMatrix)
plt.show()

# Coil Data

coil_data = dico.iloc[:,103:116]
coil_data = pd.DataFrame(coil_data).fillna(value = 0, )

coil_data_umap = umap_projection(df = coil_data, data_title= 'Coil', n_components=2)







a = "fu"+"ck"