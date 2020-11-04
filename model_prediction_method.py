#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:32:10 2020

@authors: youpele and dan
"""

import shutil
import torch
from torchvision import datasets, models, transforms
import os





    
'''
 ___________________________
|                           |
|       THE METHOD          |
|___________________________|

'''
    

def predict_signal(model_path, containing_folder, image_folder, image_index = 1):
    
    # Setting up folders
    folders = ['predicted_corrupt', 'predicted_good' ]
    
    for folder in folders:
        new_folder = containing_folder + folder
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            print("Folder " , new_folder ,  " has been created ")
        else: 
            print("Directory " , new_folder ,  " already exists")
            
            
    
    # Prediction and copying predicted image to its respective folder
    
    model = torch.load(model_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pred_s = []
    tf = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            #transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    image_datasets = datasets.ImageFolder(containing_folder, tf)
    inputs = image_datasets[image_index]
    print (inputs[1])
    with torch.no_grad():
        inputs = inputs[0][None, ...].to(device)
        outputs = model(inputs)
        pred_s.append(outputs.argmax(dim=1).cpu().item())
    print(pred_s[0])
    
    
    image_folder_list = os.listdir(image_folder)
    if pred_s[0] == 0:
        print('This signal is corrupted!')
        
        shutil.copy(image_folder + image_folder_list[image_index],
                        containing_folder + folders[0] + '/' + image_folder_list[image_index])
        print('Image has been copied to', folders[0], ' folder.')
        
    elif pred_s[0] == 1:
        
        print('This signal is good!')
        shutil.copy(image_folder + image_folder_list[image_index],
                        containing_folder + folders[1] + '/' + image_folder_list[image_index])
        print('Image has been copied to', folders[1], 'folder.')
        
        
   
    
    
'''
 ___________________________
|                           |
|   HOW TO CALL THE METHOD  |
|___________________________|

'''
    
# single image prediction


containing_folder = '/Users/youpele/Documents/WZL/12032020/dataset/unused/'

image_folder = '/Users/youpele/Documents/WZL/12032020/dataset/unused/files/'

model_path = "/Users/youpele/Documents/WZL/12032020/another1_with_3fc_layers.pth"

image_index = 0




a = predict_signal(model_path=model_path,
                              containing_folder=containing_folder,
                              image_index=image_index, 
                              image_folder=image_folder)


# multiple image prediction

containing_folder = '/Users/youpele/Documents/WZL/12032020/dataset/unused/'

image_folder = '/Users/youpele/Documents/WZL/12032020/dataset/unused/signal_folder/'

model_path = "/Users/youpele/Documents/WZL/12032020/another1_with_4fc_layers.pth"

"/Users/youpele/Documents/WZL/12032020/anotherOneToday.pth"




'''
new_list = []
for image_index in range(0, len(os.listdir(image_folder))):
    new_list.append(image_index)

for image_index in new_list:  
    b = predict_signal(model_path=model_path,
                              containing_folder=containing_folder,
                              image_index=image_index, 
                              image_folder=image_folder)


'''


# multiple image prediction

containing_folder = "/Users/youpele/Desktop/R1_1/"

image_folder = "/Users/youpele/Desktop/R1_1/R1_1_Stempel_hinten/"

model_path = "/Users/youpele/Documents/WZL/12032020/anotherOneToday.pth"




def multi_pred_signal (model_path, containing_folder, image_folder):
    
    new_list = []
    for image_index in range(0, len(os.listdir(image_folder))):
        new_list.append(image_index)
    
    for image_index in new_list:  
        b = predict_signal(model_path=model_path,
                                  containing_folder=containing_folder,
                                  image_index=image_index, 
                                  image_folder=image_folder)
    
    
    
    
    
trying = multi_pred_signal(model_path=model_path, 
                           containing_folder=containing_folder,
                           image_folder=image_folder)

    
    
    
    