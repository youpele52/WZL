#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 06:50:34 2020

@author: youpele
"""
import os
import shutil


'''
Corrupted Files
'''
newlist = []

sorted_corrupted = []

corrupted = os.listdir('/Users/youpele/Documents/WZL/12032020/AKS-19-KS-Stempel-hinten/R3_2_Stempel_hinten/')

base = 'R3_2_Stempel_hinten_'
ext = '.png'

for i in range(2459,4859): 
    newlist.append(i)

for corrupts in corrupted:
    for i in newlist:
        if corrupts == base + str(i) + ext:
            newPath = shutil.copy(corrupts, '/Users/youpele/Documents/WZL/12032020/dataset/training_set/corrupted_data/')
            #sorted_corrupted.append(corrupts)
            
            
            
'''
 Good files
'''                 
            
newlist = []

sorted_good = []

good = os.listdir('/Users/youpele/Documents/WZL/12032020/AKS-19-KS-Stempel-hinten/R4_1_Stempel_hinten/')

base = 'R4_1_Stempel_hinten_'
ext = '.png'

for i in range(1,4001): 
    newlist.append(i)

for file in good:
    for i in newlist:
        if file == base + str(i) + ext:
            newPath = shutil.copy(file, '/Users/youpele/Documents/WZL/12032020/dataset/training_set/good_data/')
            #sorted_corrupted.append(corrupts)            
            



'''
Balancing number of datasets
'''


good_data = os.listdir('/Users/youpele/Documents/WZL/12032020/dataset/training_set/good_data/')

newlist = []

for i in range(4001,4400):
    newPath = shutil.move(good_data[i],'/Users/youpele/Documents/WZL/12032020/dataset/unused/good/' )
    newlist.append(good_data[i])






corrupt_data = os.listdir('/Users/youpele/Documents/WZL/12032020/dataset/training_set/corrupted_data')

newlist = []

for i in range(4000,4043):
    newPath = shutil.move(corrupt_data[i],'/Users/youpele/Documents/WZL/12032020/dataset/unused/corrupted/' )
    newlist.append(corrupt_data[i])




'''
Moving items to the test folders
'''

import random

for i in random.sample(os.listdir('/Users/youpele/Documents/WZL/12032020/dataset/unused/corrupted/'),
                       1200):
    shutil.move(i, '/Users/youpele/Documents/WZL/12032020/dataset/test_set/corrupted_data/')



for i in random.sample(good_data,
                       1200):
    shutil.move(i, '/Users/youpele/Documents/WZL/12032020/dataset/test_set/good_data/')




good_data[4000]
