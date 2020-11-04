

# importing lib and packages
import sqlite3
import pandas as pd
from npfeintool import FeatEx
from npfeintool import CON
from npfeintool.analyzer import Analyzer


#save the names of database files in an array
name_of_db_1 = ["DiCo_digital_coil_M1_M2_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_1, cutting_index):

    #load data into a dataframe
    df = pd.DataFrame()
    df = CON.load_all_segments("", name)
    
#only PCA
Analyzer.only_pca(df)


# For the second db file


#save the names of database files in an array
name_of_db_2 = ["DiCo_digital_coil_M2_M3_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_2, cutting_index):

    #load data into a dataframe
    df2 = pd.DataFrame()
    df2 = CON.load_all_segments("", name)

    
 #only PCA
Analyzer.only_pca(df2)   


# For the third db file


#save the names of database files in an array
name_of_db_3 = ["DiCo_digital_coil_M3_M4_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_3, cutting_index):

    #load data into a dataframe
    df3 = pd.DataFrame()
    df3 = CON.load_all_segments("", name)
    

#only PCA
Analyzer.only_pca(df3) 


# For the fourth db file


#save the names of database files in an array
name_of_db_4 = ["DiCo_digital_coil_M4_M5_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_4, cutting_index):

    #load data into a dataframe
    df4 = pd.DataFrame()
    df4 = CON.load_all_segments("", name)
    
#only PCA
Analyzer.only_pca(df4) 


# For the fifth db file


#save the names of database files in an array
name_of_db_5 = ["DiCo_digital_coil_M5_M6_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_5, cutting_index):

    #load data into a dataframe
    df5 = pd.DataFrame()
    df5 = CON.load_all_segments("", name)
    
#only PCA
Analyzer.only_pca(df5) 


# For the sixth db file


#save the names of database files in an array
name_of_db_6 = ["DiCo_digital_coil_M6_M7_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_6, cutting_index):

    #load data into a dataframe
    df6 = pd.DataFrame()
    df6 = CON.load_all_segments("", name)
    

#only PCA
Analyzer.only_pca(df6)



# For the seventh db file


#save the names of database files in an array
name_of_db_7 = ["DiCo_digital_coil_M7_M8_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_7, cutting_index):

    #load data into a dataframe
    df7 = pd.DataFrame()
    df7 = CON.load_all_segments("", name)
    
#only PCA
Analyzer.only_pca(df7) 


# For the eighth db file


#save the names of database files in an array
name_of_db_8 = ["DiCo_digital_coil_M8_M9_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_8, cutting_index):

    #load data into a dataframe
    df8 = pd.DataFrame()
    df8 = CON.load_all_segments("", name)
    

#only PCA
Analyzer.only_pca(df8)


# For the ninth db file


#save the names of database files in an array
name_of_db_9 = ["DiCo_digital_coil_M9_M10_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_9, cutting_index):

    #load data into a dataframe
    df9 = pd.DataFrame()
    df9 = CON.load_all_segments("", name)
    
#only PCA
Analyzer.only_pca(df9)


# For the tenth db file


#save the names of database files in an array
name_of_db_10 = ["DiCo_digital_coil_M10_5_M11_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [2399, 1499, 799]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_10, cutting_index):

    #load data into a dataframe
    df10 = pd.DataFrame()
    df10 = CON.load_all_segments("", name)
    

#only PCA
Analyzer.only_pca(df10)



# For the eleventh db file


#save the names of database files in an array
name_of_db_11 = ["DiCo_digital_coil_M10_M10_5_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [1500]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_11, cutting_index):

    #load data into a dataframe
    df11 = pd.DataFrame()
    df11 = CON.load_all_segments("", name)
    

#only PCA
Analyzer.only_pca(df11)


'''
Visualizing all the db files as one¶
There seems to be a problem

'''



import sqlite3
import pandas as pd
from npfeintool import FeatEx
from npfeintool import CON
from npfeintool.analyzer import Analyzer



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
cutting_index = [1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,1500,]


#iterating over all database files and extracting features for every single database
for name, index in zip(names_databases, cutting_index):

    #load data into a dataframe
    df = pd.DataFrame()
    df = CON.load_all_segments("", name)
    
    
#only PCA
Analyzer.only_pca(df)



'''
Visualising the eleventh db file
Using different filters

'''

# For the eleventh db file


#save the names of database files in an array
name_of_db_11 = ["DiCo_digital_coil_M10_M10_5_Stempel_1_punches"]

#current indices at which the segments should be cutted
cutting_index = [1500]



#iterating over all database files and extracting features for every single database
for name, index in zip(name_of_db_11, cutting_index):

    #load data into a dataframe
    df = pd.DataFrame()
    df = CON.load_all_segments("", name)
    df = df.truncate(before=0, after=1499)
    df_transpose = df.transpose()
    df_no_missing = df_transpose.dropna().values # Remove signal with missing data


# only PCA
Analyzer.only_pca(df_no_missing)

# dwt + PCA
Analyzer.dwt_pca(df_no_missing)

# DWT + staistics + PCA
Analyzer.dwt_stats_pca(df_no_missing)