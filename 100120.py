#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:53:39 2020

@author: youpele
"""


import pandas as pd
from statistics import mean


# features 

feat_M1 = pd.read_json("features_M1_M2.json")
feat_M2 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M2_M3.json")
feat_M3 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M3_M4.json")
feat_M4 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M4_M5.json")
feat_M5 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M5_M6.json")
feat_M6 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M6_M7.json")
feat_M7 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M7_M8.json")
feat_M8 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M8_M9.json")
feat_M9 = pd.read_json("/Users/youpele/Documents/WZL/final/features_M9_M10.json")
feat_M10b = pd.read_json("/Users/youpele/Documents/WZL/final/features_M10_5_M11.json")
feat_M10a = pd.read_json("/Users/youpele/Documents/WZL/final/features_M10_M10_5.json")


#feat = [feat_M1, feat_M2, feat_M3, feat_M4, feat_M5, feat_M6, feat_M7, feat_M8, feat_M9, feat_M10a, feat_M10b]
feat_Ms_merged = pd.concat([feat_M1, feat_M2, feat_M3, feat_M4, feat_M5, feat_M6, feat_M7, feat_M8, feat_M9, feat_M10a, feat_M10b], 
                      ignore_index=True)

feat_Ms_merged['Punch'] = feat_Ms_merged.index


# Way to punch

way_to_punch = pd.read_json("/Users/youpele/Documents/WZL/final/way_to_punch.json")

way_to_punch['Punch'] = way_to_punch.index


# WZL 

thickness_wzl = pd.read_json("/Users/youpele/Documents/WZL/final/thickness_wzl.json")

# ibf
thickness_ibf = pd.read_json("/Users/youpele/Documents/WZL/final/thickness_ibf.json")



# Merging feat & way to punch


feat_way = pd.merge(left = feat_Ms_merged, right = way_to_punch, 
                      on= 'Punch', how = 'outer')

# Merging feat_way & wzl

feat_way_wzl = pd.merge(left = feat_way, right = thickness_wzl, 
                      on= 'way', how = 'outer')


for i in range(len(thickness_ibf)):

    if thickness_wzl.loc[i,'way'] == thickness_ibf.loc[i,'way']:
        print(thickness_ibf.loc[i,'way'])





for i in range(len(thickness_ibf)):
    
    if thickness_wzl[i]['way'] == thickness_ibf[i]['way']:
        print(i)



wzl_ibf = pd.merge(left = thickness_wzl, right = thickness_ibf, how = 'outer',
                             left_index = True, right_index = True)

new_list = []
for i in range(len(wzl_ibf)):

if wzzzz = print(wzl_ibf.loc[0:18851,'way_x'] == wzl_ibf.loc[0:18851,'way_y']) :
    print(wzl_ibf.loc[0:18851,'way_x'])
    
    
aaaalist = []
for i in wzl_ibf.loc[0:18851,'way_x']:
    for j in wzl_ibf.loc[0:18851,'way_y']:
        if j==i:
            aaaalist.append(j)
            
        
        
        if i-0.2 <= j >= i+0.2:
            aaaalist.append(j)
        
        j:
            print(i)
            aaaalist.append(i)
            
    
wzzz = wzl_ibf.loc[0,'way_y']

for i in wzzz:
    print (i)
    
    
a = range(0,50)
alist=[]
b = range(25,75)
blist=[]

for i in a:
    alist.append(i)

for i in b:
    blist.append(i)
    
    
for i in blist:
    for j in alist:
        if i == j:
            print(i)
    
    alist[i] == blist [i]
    print(i)














if feat_way_wzl.loc[]






for i in feat_way_wzl[102][102]:
    print(i)

feat_way_wzl.iloc[1,101:102]
