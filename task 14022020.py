#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:53:58 2020

@author: youpele
"""

#imports
import umap
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def umap_projection (df, data_title = "", n_components = 2 ):
    
    '''
    
    This function reduces a multi feature datasets to a datasets with n_components using umap. 
    It accepts a dataframe, data title and n_components.
    '''
    
    
    embedding = umap.UMAP(n_neighbors=15,
                      metric='correlation', 
                      n_components = n_components).fit_transform(df)
    
    if n_components == 2:
        plt.scatter(embedding[:, 0], embedding[:, 1],)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title('UMAP projection of the {a} data.'.format(a = data_title))
        plt.savefig(fname = data_title,dpi=1200)
    
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




# Coil Data

coil_data = dico.iloc[:,103:116]
coil_data = pd.DataFrame(coil_data).fillna(value = 0, )

entry_force = coil_data.iloc[:, 0:4]
exit_force = coil_data.iloc[:, 4:8]
thickness = coil_data.iloc[:, 8:12]


'''
UMAP Projection 
'''

# Gegenhalter
Gegenhalter_umap = umap_projection(df = Gegenhalter, data_title= 'Gegenhalter')


#Niederhalter
Niederhalter_umap = umap_projection(df = Niederhalter, data_title= 'Niederhalter')

Niederhalter_1_umap = umap_projection(df = Niederhalter_1, data_title= 'Niederhalter_1')

Niederhalter_2_umap = umap_projection(df = Niederhalter_2, data_title= 'Niederhalter_2')

Niederhalter_4_umap = umap_projection(df = Niederhalter_4, data_title= 'Niederhalter_4')



# Stempel
Stempel_umap = umap_projection(df = Stempel, data_title= "Stempel")

Stempel_1_umap = umap_projection(df = Stempel_1, data_title= "Stempel_1")

Stempel_2_umap = umap_projection(df = Stempel_2, data_title= "Stempel_2")

Stempel_3_umap = umap_projection(df = Stempel_3, data_title= "Stempel_3")

Stempel_4_umap = umap_projection(df = Stempel_4, data_title= "Stempel_4")


# coil Data
coil_data_umap = umap_projection(df = coil_data, data_title= 'Coil', n_components=2)

entry_force_umap = umap_projection(df = entry_force, data_title= 'entry_force', n_components=2)

exit_force_umap = umap_projection(df = exit_force, data_title= 'exit_force', n_components=2)

thickness_umap = umap_projection(df = thickness, data_title= 'thickness', n_components=2)


#Gegenhalter + Niederhalter + Stempel
Ge_Nie_Stem = pd.concat([Gegenhalter, Niederhalter, Stempel], axis=1)

Ge_Nie_Stem_umap_10 = umap_projection(df = Ge_Nie_Stem, 
                                   data_title= 'Gegenhalter + Niederhalter + Stempel',n_components=10)

Ge_Nie_Stem_umap_2 = umap_projection(df = Ge_Nie_Stem, 
                                   data_title= 'Gegenhalter + Niederhalter + Stempel',n_components=2)




'''
Correlation Matrices with coil data
'''


# Correlation Matrices


class CorrMatrix():
        
        
    def corrMatrix(df_1, df_2, data_title = "", excel_export = "no"):
    
        """
        This function accepts two dataframes, merge them and returns a correlation matrix.
        """
        
        joining = pd.concat([df_1, df_2], axis =1)
        corrMatrix = joining.corr()
    
        
        
        name_1 =[x for x in globals() if globals()[x] is df_1][0]
        name_2 =[x for x in globals() if globals()[x] is df_2][0]  
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corrMatrix, interpolation='nearest')
        plt.title('Correlation Matrix of {a} and {b}.'.format(a= name_1 , b = name_2 ))
        fig.colorbar(cax)
        plt.savefig(fname = data_title,dpi=1200)
        if excel_export == "yes":
            corrMatrix.to_excel(str(data_title)+".xlsx")
        
        return corrMatrix


    def multiple_corrMatrix(df, data_title):
        
        """
        This function accepts df (preferably, umap_ed df) and returns a 
        correlation matrix of the inputed df, the coil data and each data in it, that is 
        coil_data_umap,entry_force_umap, exit_force_umap, thickness_umap.
        """
        for n, i in enumerate ([coil_data_umap,entry_force_umap, exit_force_umap, thickness_umap]):
            
            a = corrMatrix(df_1 = df, df_2 = i,
                           excel_export="no", data_title = data_title+"_corr" +'%s' % str(n + 1))





# Gegenhalter 
            
CorrMatrix.multiple_corrMatrix( df = Gegenhalter_umap, data_title="Gegenhalter")


#Niederhalter

CorrMatrix.multiple_corrMatrix(df = Niederhalter_umap, data_title="Niederhalter")

CorrMatrix.multiple_corrMatrix(df = Niederhalter_1_umap, data_title="Niederhalter_1")

CorrMatrix.multiple_corrMatrix(df = Niederhalter_2_umap, data_title="Niederhalter_2")

CorrMatrix.multiple_corrMatrix(df = Niederhalter_4_umap, data_title="Niederhalter_4")



#Stempel
CorrMatrix.multiple_corrMatrix(df = Stempel_umap, data_title="Stempel")


CorrMatrix.multiple_corrMatrix(df = Stempel_1_umap, data_title="Stempel_1")

CorrMatrix.multiple_corrMatrix(df = Stempel_2_umap, data_title="Stempel_2")

CorrMatrix.multiple_corrMatrix(df = Stempel_3_umap, data_title="Stempel_3")

CorrMatrix.multiple_corrMatrix(df = Stempel_4_umap, data_title="Stempel_4")


    
    
# Ge_Nie_Stem_umap_2

CorrMatrix.multiple_corrMatrix(df = Ge_Nie_Stem_umap_2, data_title="Ge_Nie_Stem_umap")


# Ge_Nie_Stem_umap_10

CorrMatrix.multiple_corrMatrix(df = Ge_Nie_Stem_umap_10, data_title="Ge_Nie_Stem_umap")


