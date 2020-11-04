
import sqlite3
import pandas as pd
from npfeintool import FeatEx
from npfeintool import CON


#save the names of database files in an array, important if you want to change them without .db at the end
names_databases = ["DiCo_digital_coil_M1_M2_Stempel_1_punches", 
                    "DiCo_digital_coil_M2_M3_Stempel_1_punches", 
                    "DiCo_digital_coil_M3_M4_Stempel_1_punches",
                    "DiCo_digital_coil_M4_M5_Stempel_1_punches",
                    "DiCo_digital_coil_M5_M6_Stempel_1_punches",
                    "DiCo_digital_coil_M6_M7_Stempel_1_punches",
                    "DiCo_digital_coil_M7_M8_Stempel_1_punches",
                    "DiCo_digital_coil_M8_M9_Stempel_1_punches",
                    "DiCo_digital_coil_M9_M10_Stempel_1_punches",
                    "DiCo_digital_coil_M10_5_M11_Stempel_1_punches",
                    "DiCo_digital_coil_M10_M10_5_Stempel_1_punches",
                    ]


#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]


#iterating over all database files and extracting features for every single database
for name, index in zip(names_databases, cutting_index):

    #load data into a dataframe
    df = pd.DataFrame()
    df = CON.load_all_segments("", name)



# Feature Scaling # feature scaling is also important for pca
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_scaled = sc.fit_transform(df)




 # Applying PCA
from sklearn.decomposition import PCA # inmporting the lib of pca
pca = PCA(n_components = 2)
projected = pca.fit_transform(df_scaled)



# Visualize 2D Projection
import matplotlib
import matplotlib.pyplot as plt


plt.scatter(projected[:, 0], projected[:, 1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid()


