#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:30:08 2020

@author: youpele
"""

#imports
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dico = pd.read_json("dico_features.json")

dico_2 = dico.drop(['way', 'index', 'segment'], axis=1)


# correlation matrix

corrMatrix = dico_2.corr()


#plots


plt.matshow(corrMatrix)
plt.show()


# larger plots

f = plt.figure(figsize=(19, 15))
plt.matshow(corrMatrix, fignum=f.number)
plt.xticks(range(corrMatrix.shape[1]), corrMatrix.columns, fontsize=14, rotation=45)
plt.yticks(range(corrMatrix.shape[1]), corrMatrix.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);


corrMatrix.style.background_gradient(cmap='coolwarm').set_precision(2)


corrMatrix.to_excel("Corr Matrix.xlsx")