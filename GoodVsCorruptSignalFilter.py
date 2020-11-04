#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 16:36:28 2020

@author: youpele
"""



from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import shutil
import pandas as pd

#from PIL import Image


class Filter:
    
             
    @staticmethod
    def create_model(data_dir, work_dir, num_epochs=25):
        '''
        Creates a model that can be differentiate between good and corrupted segmentations.
        
        data_dir: The path to the folder, which should contain two other folders, train and val. The train and val folders should each contain the two folders; one containing the corrupted segmentations and the other the good segmentations.
        
        work_dir: The folder path in which the created model would be saved.
        
        num_epochs: The number of epoch.
        
        '''
        
        # Data augmentation and normalization for training
        # Just normalization for validation
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        
        #data_dir = '/Users/youpele/Documents/WZL/12032020/dataset'
        
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=100, # reduce the batch to 2-4 if you want to images displayed in the method below
                                                     shuffle=True, num_workers=0)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        
        class_names = image_datasets['train'].classes
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        # Visualize few images
        #@classmethod
        def imshow(inp, title=None):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated
        
        
        
        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))
        
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        
        show = imshow(out, title=[class_names[x] for x in classes])
        
        
        
        

        #  Training the model
        
        def train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs):
            since = time.time()
        
            best_model_wts = copy.deepcopy(model.state_dict()) # saving the best model
            best_acc = 0.0
        
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)
        
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode
        
                    running_loss = 0.0
                    running_corrects = 0
        
                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)
        
                        # zero the parameter gradients
                        optimizer.zero_grad()
        
                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)
        
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
        
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()
        
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))
        
                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
        
                print()
        
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))
        
            # load best model weights
            model.load_state_dict(best_model_wts)
            return model  
        
        
        #Finetuning the convnet
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features # getting the features to use in the layers we shall add
        # reconstructing the last fc layer
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        
        model_ft.fc = nn.Linear(num_ftrs, 1024)
        model_ft.fc1 = nn.Linear(1024,2048)
        model_ft.fc2 = nn.Linear(2048,4096)
        model_ft.fc3 = nn.Linear(4096,2)
        
        
        
        model_ft = model_ft.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
        
        
        # Training
        
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs = num_epochs)
        
 
        #Visualizing the model predictions
        
        def visualize_model(model, num_images=6):
            was_training = model.training
            model.eval()
            images_so_far = 0
            fig = plt.figure()
        
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(dataloaders['val']):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
        
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
        
                    for j in range(inputs.size()[0]):
                        images_so_far += 1
                        ax = plt.subplot(num_images//2, 2, images_so_far)
                        ax.axis('off')
                        ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                        imshow(inputs.cpu().data[j])
        
                        if images_so_far == num_images:
                            model.train(mode=was_training)
                            return
                model.train(mode=was_training)
            
            
        # Visuals
        visualize_model(model_ft)
        
        # saving the model
        
        torch.save(model_ft, work_dir + "/GoodVsCorruptModel.pth")
        
        # loading the model, use the code below
        
        #model = torch.load( work_dir + "/GoodVsCorruptModel.pth") 



    # Predict Segment
    
    def predict_signal(model_path, containing_folder, image_folder, image_index = 1):
        
        '''
        This method predicts a single segmentation(punch) whether it is corrupt or not.
        
        model_path: Path to the model.
        
        image_folder: Folder in which the segmentation is.
        
        containing_folder: Folder in which the image_folder is in. 
        
        
        '''
        
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
            
            
            
    
    def multi_pred_signal (model_path, containing_folder, image_folder):
        
        '''
        This method predicts  multiple segmentations(punches) whether it is corrupt or not.
        
        model_path: Path to the model.
        
        image_folder: Folder in which the segmentations are.
        
        containing_folder: Folder in which the image_folder is in. 
        
        
        '''
        
        new_list = []
        for image_index in range(0, len(os.listdir(image_folder))):
            new_list.append(image_index)
        
        for image_index in new_list:  
            b = Filter.predict_signal(model_path=model_path,
                                      containing_folder=containing_folder,
                                      image_index=image_index, 
                                      image_folder=image_folder)
            
            
    
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

        
    
    
    def corrupt_punch_check2(filepathed___):
        
        '''
        Computes information such number of corrupt segmentations/punches and their respective index numbers for a multiple revisions. 
        
        filepathed___: File path of the folder containing multiple revisions.
        
        '''
    
        info_list = []
        #date_ = str(date.today()) 
        #date_ = date_.replace("-", "_")
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
        
    
    


        
        
   
    