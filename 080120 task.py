#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:26:29 2020

@author: youpele
"""


import pandas as pd
from matplotlib import pyplot as plt
from statistics import mean

df_M1 = pd.read_json("features_M1_M2.json")
df_M2 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M2_M3.json")
df_M3 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M3_M4.json")
df_M4 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M4_M5.json")
df_M5 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M5_M6.json")
df_M6 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M6_M7.json")
df_M7 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M7_M8.json")
df_M8 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M8_M9.json")
df_M9 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M9_M10.json")
df_M10b = pd.read_json("/Users/youpele/Documents/WZL/final/features_M10_5_M11.json")
df_M10a = pd.read_json("/Users/youpele/Documents/WZL/final/features_M10_M10_5.json")


#df = [df_M1, df_M2, df_M3, df_M4, df_M5, df_M6, df_M7, df_M8, df_M9, df_M10a, df_M10b]
df_merged = pd.concat([df_M1, df_M2, df_M3, df_M4, df_M5, df_M6, df_M7, df_M8, df_M9, df_M10a, df_M10b], 
                      ignore_index=True)


df_merged_Stempel = df_merged.loc[:, ['max_Stempel_1','min_Stempel_1', 'mean_Stempel_1', 
                                      'max_Stempel_2','min_Stempel_2', 'mean_Stempel_2',
                                      'max_Stempel_3','min_Stempel_3', 'mean_Stempel_3', 
                                      'max_Stempel_4','min_Stempel_4', 'mean_Stempel_4']]


way_to_punch = pd.read_json("/Users/youpele/Documents/WZL/final/way_to_punch.json")


thickness_wzl_df = pd.read_json("/Users/youpele/Documents/WZL/final/thickness_wzl.json")
thickness_wzl_num_thick = thickness_wzl_df.loc[:, ['Number','thickness']]


thickness_ibf_df = pd.read_json("/Users/youpele/Documents/WZL/final/thickness_ibf.json")
thickness_ibf_force = thickness_ibf_df.loc[:, ['entryCoiler_force', 'exitCoiler_force', 'thickness']]





data_digital_coil1 = pd.merge(left = way_to_punch, right = thickness_ibf_force,
                             left_index = True, right_index = True)

data_digital_coil2 = pd.merge(left = data_digital_coil1, right = thickness_wzl_num_thick,
                             left_index = True, right_index = True)


data_digital_coil3 = pd.merge(left = data_digital_coil2, right = df_merged_Stempel,
                             left_index = True, right_index = True)

data_digital_coil = data_digital_coil3


data_digital_coil_export = data_digital_coil.to_json(r'/Users/youpele/Documents/WZL/final/080120_task.json')


dfffffff= pd.read_json("080120_task.json")



thickness_ibdway= thickness_ibf_df.loc[:, ['way']]
thickforce = thickness_ibf_df.loc[:, ['exitCoiler_force']]

plt.plot(thickness_ibdway, thickforce)




row = len(thickness_ibf_df)


ls = []
for s in range(0,2693):
    ls.append(int(row/2693))
    
for s in ls:
    df_temp = thickness_ibf_df.iloc[s:int(s+ls[1])]
    


mean([1,2,3])




def df_split (df, d = 4): 
    
    '''
    Takes in dataframe and a depth and split the dataframe according to the depth number inputted

    '''
    
    row = len(df)

    list_dfs = []
    
    ls = []
    for s in range(0,d):
        ls.append(int(s/d*row))

    for s in ls:
        df_temp = df.iloc[s:int(s+ls[1])]
        #arr = np.array(df_temp)
        #df_temp = pd.DataFrame(data=arr.flatten())
        list_dfs.append(df_temp)
    
    return list_dfs




avg_entry_force = []
avg_exit_force = []
avg_thickness = []


thicknesss_ibf = df_split(thickness_ibf_df,2693)


avg_exit_force.append(mean(thicknesss_ibf[i]['exitCoiler_force']))


'entryCoiler_force', 'exitCoiler_force', 'thickness'

for i in range(len(thicknesss_ibf)):
    avg_entry_force.append(mean(thicknesss_ibf[i]['entryCoiler_force']))
    avg_exit_force.append(mean(thicknesss_ibf[i]['exitCoiler_force']))
    avg_thickness.append(mean(thicknesss_ibf[i]['thickness']))
    

thickness_avg_ibf = pd.DataFrame({"avg_entry_force":avg_entry_force,
                        "avg_exit_force": avg_exit_force,
                        "avg_thickness": avg_thickness})


    
    

























