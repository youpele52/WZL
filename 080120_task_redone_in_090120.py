#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 11:52:55 2020

@author: youpele
"""


import pandas as pd
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
df_Ms_merged = pd.concat([df_M1, df_M2, df_M3, df_M4, df_M5, df_M6, df_M7, df_M8, df_M9, df_M10a, df_M10b], 
                      ignore_index=True)

df_Ms_merged['Punch'] = df_Ms_merged.index



local_M1 = pd.read_json("local_globa_M1.json")
local_M2 = pd.read_json("local_globa_M2.json")
local_M3 = pd.read_json("local_globa_M3.json")
local_M4 = pd.read_json("local_globa_M4.json")
local_M5 = pd.read_json("local_globa_M5.json")
local_M6 = pd.read_json("local_globa_M6.json")
local_M7 = pd.read_json("local_globa_M7.json")
local_M8 = pd.read_json("local_globa_M8.json")
local_M9 = pd.read_json("local_globa_M9.json")
local_M10b = pd.read_json("local_globa_M10_5.json")
local_M10a = pd.read_json("local_globa_M10.json")


local_Ms_merged =  pd.concat([local_M1, local_M2, local_M3, local_M4, local_M5, local_M6, 
                              local_M7, local_M8, local_M9, local_M10a, local_M10b], 
                      ignore_index=True)


local_M1


way_to_punch = pd.read_json("/Users/youpele/Documents/WZL/final/way_to_punch.json")

way_to_punch['Punch'] = way_to_punch.index






thickness_wzl_df = pd.read_json("/Users/youpele/Documents/WZL/final/thickness_wzl.json")



thickness_wzl_num = thickness_wzl_df.loc[:, ['Number']]


thickness_ibf_df = pd.read_json("/Users/youpele/Documents/WZL/final/thickness_ibf.json")



# Using split func



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



thicknesss_ibf = df_split(thickness_ibf_df,2693)



avg_entry_force = []
avg_exit_force = []
avg_thickness = []


# Averaging

for i in range(len(thicknesss_ibf)):
    avg_entry_force.append(mean(thicknesss_ibf[i]['entryCoiler_force']))
    avg_exit_force.append(mean(thicknesss_ibf[i]['exitCoiler_force']))
    avg_thickness.append(mean(thicknesss_ibf[i]['thickness']))
    

# Creating a new dataframe containing the averages of entry/exit force and thickness
thickness_avg_ibf = pd.DataFrame({"avg_entry_force":avg_entry_force,
                        "avg_exit_force": avg_exit_force,
                        "avg_thickness": avg_thickness})





# Merging the dataframes

data_digital_coil1 = pd.merge(left = way_to_punch, right = thickness_avg_ibf,
                             left_index = True, right_index = True)

data_digital_coil2 = pd.merge(left = data_digital_coil1, right = thickness_wzl_num,
                             left_index = True, right_index = True)


data_digital_coil3 = pd.merge(left = data_digital_coil2, right = df_merged,
                             left_index = True, right_index = True)


data_digital_coil = data_digital_coil3


#Exporting the created dataframe to json

data_digital_coil_export = data_digital_coil.to_json(r'/Users/youpele/Documents/WZL/final/080120_task_redone.json')



# Reading the newly created json


read_new_json= pd.read_json("080120_task_redone.json")







sjsjgssjdsdg=pd.merge(left = way_to_punch, right = thickness_wzl_df, 
                      on= 'way')


ssssyesyes = pd.merge(left = way_to_punch, right = thickness_wzl_df, how = 'outer',
                      on= 'way')





local_global = pd.read_json("/Users/youpele/Documents/WZL/final/local_globa_M2.json")