#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 12:56:26 2020

@author: youpele
"""

import os
import cv2
import numpy as np
from tqdm import tqdm # this is progress bar basically
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




REBUILD_DATA = True



class GoodVSCorrupt():
    IMG_SIZE = 50
    CORRUPT = "/Users/youpele/Documents/WZL/12032020/corrupt_vs_good_signal/corrupted_data/"
    GOOD = "/Users/youpele/Documents/WZL/12032020/corrupt_vs_good_signal/good_data/"
    TESTING = "PetImages/Testing"
    LABELS = {CORRUPT: 0, GOOD: 1}
    training_data = []

    cor_count = 0
    go_count = 0

    def make_training_data(self):
        for label in self.LABELS:
            print(label)
            for f in tqdm(os.listdir(label)):
                if "png" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img), np.eye(2)[self.LABELS[label]]])  # do something like print(np.eye(2)[1]), just makes one_hot 
                        
                        # np.eye(2)[self.LABELS[label]] 
                        # this line create one hot encode for the classes
                        # if its is cat it returns [1,0], and for dog [0,1]
                        #print(np.eye(2)[self.LABELS[label]])

                        if label == self.CORRUPT:
                            self.cor_count += 1
                        elif label == self.GOOD:
                            self.go_count += 1

                    except Exception as e:
                        pass
                        #print(label, f, str(e))

        np.random.shuffle(self.training_data)
        np.save("training_data.npy", self.training_data)
        print('CORRUPT:',goodvscorrupt.cor_count)
        print('GOOD:',goodvscorrupt.go_count)
        
        
        
        
        
        


# creating the model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        # input = 1
        # output = 32  convolutional features
        # kernel = 5, this is going to make a 5 by 5 kernel/window as it rolls/slides over our data to features
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=5)
        self.conv4 = nn.Conv2d(128, 256, 5)
        # nn.Conv3d are used for scans or models

        x = torch.randn(50,50).view(-1,1,50,50)
        # 50*50 is the size of the image that we have resized earlier
        # view to flatten the image, 
        # (-1, to prepare to accept any feature we have
        # 1, a mirror of the input in the first conv1 layer
        # 50*50, same as before. )
        
        self._to_linear = None
        
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 512)
        # the 512 is random guessed number
        # we dont know the number that is supposed to be in self._to_linear position 
        # thus theres  a lil scrip forward that will allow us predict then use it automatically 
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # 2 by 2, is the shape of the pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        x = F.max_pool2d(F.relu(self.conv4(x)), (2,2))
        

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)  #we flatten
        x = F.relu(self.fc1(x))
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)



# check for gpu 
        
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")



net = Net().to(device)

if REBUILD_DATA:
    goodvscorrupt = GoodVSCorrupt()
    goodvscorrupt.make_training_data()
        
    
    
    
    


training_data = np.load("training_data.npy", allow_pickle=True)
print(len(training_data))

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

X = torch.Tensor([i[0] for i in training_data]).view(-1, 50, 50)
X = X/255.0
y = torch.Tensor([i[1] for i in training_data])

VAL_PCT = 0.1
val_size = int(len(X)*VAL_PCT)
print(val_size)

train_X = X[:-val_size]
train_y = y[:-val_size]

test_X = X[-val_size:]
test_y = y[-val_size:]

print(len(train_X))
print(len(test_X))








# This returns the train data accuracy and loss

def fwd_pass(X, y, train=False):
    # when data pass thru here by default weight will not be updated

    if train:
        net.zero_grad()
    outputs = net(X)
    matches  = [torch.argmax(i)==torch.argmax(j) for i, j in zip(outputs, y)]
    acc = matches.count(True)/len(matches)
    loss = loss_function(outputs, y)

    if train:
        loss.backward() # backpropagation
        optimizer.step()

    return acc, loss




# This returns the test data accuracy and loss
    
def test(size=32):
    X, y = test_X[:size], test_y[:size]
    val_acc, val_loss = fwd_pass(X.view(-1, 1, 50, 50).to(device), y.to(device))
    return val_acc, val_loss

val_acc, val_loss = test(size=1000)
print(val_acc, val_loss)






import time

MODEL_NAME = f"model-{int(time.time())}"  # gives a dynamic model name, to just help with things getting messy over time. 
net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()

print(MODEL_NAME)


def train(net):
    BATCH_SIZE = 100
    EPOCHS = 3 #30

    with open("model.log", "a") as f:
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
                batch_X = train_X[i:i+BATCH_SIZE].view(-1,1,50,50)
                batch_y = train_y[i:i+BATCH_SIZE]

                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                acc, loss = fwd_pass(batch_X, batch_y, train=True)

                #print(f"Acc: {round(float(acc),2)}  Loss: {round(float(loss),4)}")
                #f.write(f"{MODEL_NAME},{round(time.time(),3)},train,{round(float(acc),2)},{round(float(loss),4)}\n")
                # just to show the above working, and then get out:
                if i % 50 == 0:
                    # for every 50 steps we;ll calc val accuracy and loss
                    val_acc, val_loss = test(size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss), 4)},{round(float(val_acc),2)},{round(float(val_loss),4)},{epoch}\n")




train(net)





# Comparing test accu, loss vs train accu, loss
# plot this on sublime 



import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

#model_name = MODEL_NAME #"model-1570499409" # grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.

model_name = "model-1584708342"

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, epoch = c.split(",")

            times.append(float(timestamp))
            accuracies.append(float(acc))
            losses.append(float(loss))

            val_accs.append(float(val_acc))
            val_losses.append(float(val_loss))


    fig = plt.figure()

    ax1 = plt.subplot2grid((2,1), (0,0)) #(2,1) 2 by 1 grid, (0,0) this graph will start at the x=0,y=0 mark
    ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1) # (1,0) start at 1,0 mark, and will share x axis with  graph ax1


    ax1.plot(times, accuracies, label="acc")
    ax1.plot(times, val_accs, label="val_acc")
    ax1.legend(loc=2) # location 2
    ax2.plot(times,losses, label="loss")
    ax2.plot(times,val_losses, label="val_loss")
    ax2.legend(loc=2)
    plt.show()

create_acc_loss_graph(model_name)


















    
    
    
    
    
    
    
        
        
