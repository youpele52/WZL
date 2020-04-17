#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:46:26 2020

@author: youpele
"""

import SimpleITK as sitk
import pandas as pd
import os



# Getting the Spacing, Size and Origin  info from each case 

files = os.listdir("/Volumes/Youpele_HD/Uniklinik/kits19/data/")
image_df = []
segmentation_df = []

for file in files:
    if "case" in file:
        filepath = "/Volumes/Youpele_HD/Uniklinik/kits19/data/" + file + "/"
        try:
            image = sitk.ReadImage(filepath + "imaging.nii.gz")
            image_origin = image.GetOrigin()
            image_size = image.GetSize()
            image_spacing = image.GetSpacing()
            data_image = pd.DataFrame({'Case': file,
                          'Origin_0':image_origin[0],
                          'Origin_1':image_origin[1],
                          'Origin_2':image_origin[2],
                          
                          'Size_0': image_size[0],
                          'Size_1': image_size[1],
                          'Size_2': image_size[2],
                          
                          'Spacing_0': image_spacing[0],
                          'Spacing_1': image_spacing[1], 
                          'Spacing_2': image_spacing[2],
                }, index = [0])
            	
            image_df.append(data_image)
            
        except:
            print(file + " does not have imaging.nii.gz file. ")
            
        
        
        
        try:
            segmentation = sitk.ReadImage(filepath + "segmentation.nii.gz")
            segmentation_origin = segmentation.GetOrigin()
            segmentation_size = segmentation.GetSize()
            segmentation_spacing = segmentation.GetSpacing()
            data_segmentation =pd.DataFrame({'Case': file,
                          'Origin_0':segmentation_origin[0],
                          'Origin_1':segmentation_origin[1],
                          'Origin_2':segmentation_origin[2],
                          
                          'Size_0': segmentation_size[0],
                          'Size_1': segmentation_size[1],
                          'Size_2': segmentation_size[2],
                          
                          'Spacing_0': segmentation_spacing[0],
                          'Spacing_1': segmentation_spacing[1], 
                          'Spacing_2': segmentation_spacing[2],
                }, index = [0])
            	
            segmentation_df.append(data_segmentation)
            
        except:
            print(file + " does not have segmentation.nii.gz file.")
            
        
        
            
# Concatenating the generated data of each case in the image_df list to create one merged dataframe.

result_image_df = pd.concat(image_df)

result_segmentation_df = pd.concat(segmentation_df)  


# Writing an excel file 

with pd.ExcelWriter('/Volumes/Youpele_HD/Uniklinik/Tasks/09042020.xlsx') as writer:
    result_image_df.to_excel(writer, sheet_name='Image_info')
    result_segmentation_df.to_excel(writer, sheet_name='Segmentation_info')


        



