#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 13:24:34 2020

@author: youpele
"""

# vgg16 model used for transfer learning on the dogs and cats dataset
import sys
from matplotlib import pyplot
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping


# define cnn model
def define_model():
	# load model
	model = VGG16(include_top=False, input_shape=(224, 224, 3))
	# mark loaded layers as not trainable
	for layer in model.layers:
		layer.trainable = False
	# add new classifier layers
	flat1 = Flatten()(model.layers[-1].output)
    
    
    
	class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
	output = Dense(1, activation='sigmoid')(class1)
	# define new model
	model = Model(inputs=model.inputs, outputs=output)
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
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
    train_datagen.mean = [123.68, 116.779, 103.939]
	# prepare iterator
    train_it = train_datagen.flow_from_directory('/Users/youpele/Documents/WZL/12032020/dataset/training_set/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
    test_it = test_datagen.flow_from_directory('/Users/youpele/Documents/WZL/12032020/dataset/test_set/',
		class_mode='binary', batch_size=64, target_size=(224, 224))
    
    
    checkpoint = ModelCheckpoint("/Users/youpele/Documents/WZL/12032020/corrupt_good.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)
    
    earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
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
    
    model.save("/Users/youpele/Documents/WZL/12032020/corrupt_good.h5")
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
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# center pixel data
	img = img.astype('float32')
	img = img - [123.68, 116.779, 103.939]
	return img
 
# load an image and predict the class
def run_example(image):
	# load the image
    img = load_image(image)
	# load model
    model = load_model('/Users/youpele/Documents/WZL/12032020/first_model.h5')
	# predict the class
    result = model.predict(img)
    print(result[0])
    
    
    if result[0]==0:
        print ('This signal is corrupted')      
    elif result [0] ==1:
        print('This signal is good')
    
    





#	print(result[0])
    
    
    
 
# entry point, run the example
run_example(image= '/Users/youpele/Documents/WZL/12032020/dataset/test_set/corrupted_data/R3_1_Stempel_hinten_3403.png')




