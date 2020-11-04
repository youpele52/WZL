#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 18:23:52 2020

@author: youpele
"""

import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout



# define cnn model
def define_model():
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
	# compile model
    opt = Adam(lr=0.001)#)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    return model


# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
    
    
    
  
# run the test harness for evaluating a model
def run_test_harness():
    # define model
    model = define_model()
	# create data generator
    train_datagen = ImageDataGenerator(featurewise_center=True,
                              rescale=1./255,
                              rotation_range = 20,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode = 'nearest')
    
    test_datagen = ImageDataGenerator(rescale = 1./255)
	# specify imagenet mean values for centering
    #train_datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
    train_it = train_datagen.flow_from_directory('/Users/youpele/Documents/WZL/12032020/dataset/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
    test_it = test_datagen.flow_from_directory('/Users/youpele/Documents/WZL/12032020/dataset/val/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
    
    
    checkpoint = ModelCheckpoint("/Users/youpele/Documents/WZL/12032020/second_model.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
    
    earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 4,
                          verbose = 1,
                          restore_best_weights = True)
    
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                              factor = 0.2,
                              patience = 3,
                              verbose = 1,
                              min_delta = 0.00001)
    
    # we put our call backs into a callback list
    callbacks = [earlystop, checkpoint, reduce_lr]

	# fit model
    history = model.fit_generator(train_it,
                               steps_per_epoch=len(train_it),
                               validation_data=test_it,
                               validation_steps=len(test_it),
                               epochs=10,
                               verbose=1,
                               callbacks = callbacks)
    
    model.save("/Users/youpele/Documents/WZL/12032020/second_model.h5")
	# evaluate model
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
    print('> %.3f' % (acc * 100.0))
	# learning curves
    summarize_diagnostics(history)  


# entry point, run the test harness
run_test_harness()



# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(200, 200))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 200, 200, 3)
	# center pixel data
	img = img.astype('float32')
	#img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example(image):
	# load the image
    img = load_image(image)
	# load model
    model = load_model('/Users/youpele/Documents/WZL/12032020/second_model.h5')
	# predict the class
    result = model.predict(img)
    print(result)
    
    

 
# entry point, run the example
run_example(image= '/Users/youpele/Documents/WZL/12032020/dataset/test_set/corrupted_data/R3_1_Stempel_hinten_3398.png')


    
model = load_model('/Users/youpele/Documents/WZL/12032020/second_model.h5')

a = model.predict('/Users/youpele/Documents/WZL/12032020/dataset/unused/good/R1_1_Stempel_hinten_101.png')


