#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:03:08 2020

@author: youpele
"""

import pandas as pd
import os
from datetime import date



'''
To Be Used for Single Revision
'''
    
filepath = "/Volumes/Youpele_HD/WZL/12032020/AKS-19-KS-Stempel-hinten/R1_1/"

    
def corrupt_punch_check1(filepath):
    
    '''
    Computes information such number of corrupt segmentations/punches and their respective index numbers for a single revision. 
    
    
    filepath: File path of the revision in study.
    
    '''

    revision = filepath[-5:-1]
    punch_list= []
    files = os.listdir(filepath)
    for file in files:
        if "corrupt" in file:

            corrupt_punch_list = os.listdir(filepath + file + '/')
            for punch in corrupt_punch_list:
                if punch[0] == 'R':
                    punch_list.append(punch)
                    
    df1 = pd.DataFrame({"Revision": revision,
                       "Number of corrupt punches": len(punch_list),
                       "Corrupt punches": punch_list
                       
                       }  )
    
    df2 = pd.DataFrame({"Corrupt punches": punch_list})
    
    #df_list = [df1, df2]
    
    #result_df = pd.concat(df_list)
                    
    #return result_df

    return df1



'''
TESTING
'''        
        
a = corrupt_punch_check1(filepath)
    

'''
To Be Used for Multiple Revisions
'''

filepathed___ = "/Volumes/Youpele_HD/WZL/12032020/AKS-19-KS-Stempel-hinten/"




def corrupt_punch_check2(filepathed___):
    
    '''
    Computes information such number of corrupt segmentations/punches and their respective index numbers for a multiple revisions. 
    
    filepathed___: File path of the folder containing multiple revisions.
    
    '''

    info_list = []
    date_ = str(date.today()) 
    date_ = date_.replace("-", "_")
    filepathed___list = os.listdir(filepathed___)
    for file in filepathed___list:
        if file[0]=="R":
            filepath = filepathed___ +  file + '/'
            revision = filepath[-5:-1]
            punch_list= []
            files = os.listdir(filepath)
            
            
            for file in files:
                if "corrupt" in file:
        
                    corrupt_punch_list = os.listdir(filepath + file + '/')
                    for punch in corrupt_punch_list:
                        if punch[0] == 'R':
                            punch_list.append(punch)
                            
            df1 = pd.DataFrame({"Revision": revision,
                               "Number of corrupt punches": len(punch_list),
                               "Corrupt punches": punch_list
                               
                               }  )
            
            
            info_list.append(df1)
    return info_list





            if os.path.isfile(date_ + '_new_export.xlsx'):
                with pd.ExcelWriter(date_ + '_new_export.xlsx', engine='openpyxl' ,mode = 'a') as writer:
                    df1.to_excel(writer, sheet_name=revision)
            else:
                with pd.ExcelWriter(date_ + '_new_export.xlsx') as writer:
                    df1.to_excel(writer, sheet_name=revision)
                    
                
            
            
            
                            
            
            
            







'''
TESTING
'''        
       
           
b = corrupt_punch_check2(filepathed___)
 
    
    
    
    
    
