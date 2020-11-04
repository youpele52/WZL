#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:06:15 2020

@author: youpele
"""


# Comparing test accu, loss vs train accu, loss
# plot this on sublime 



import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")

#model_name = MODEL_NAME #"model-1570499409" # grab whichever model name you want here. We could also just reference the MODEL_NAME if you're in a notebook still.

model_name = "model-1584712119"

def create_acc_loss_graph(model_name):
    contents = open("model.log", "r").read().split("\n")

    times = []
    accuracies = []
    losses = []

    val_accs = []
    val_losses = []

    for c in contents:
        if model_name in c:
            name, timestamp, acc, loss, val_acc, val_loss, none = c.split(",")

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