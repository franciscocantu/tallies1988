

#Upload Keras modules (available at keras.io)
#I am running keras with the tensorflow backend. 
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, ZeroPadding2D
from keras.callbacks import History 
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import load_model

#os is just one of default python libraries
import os

#numpy is numeric python - basically, it is most of matlab functionality ported
#to python. things like vectors, matrices, linear algebra, integrals, etc.
import numpy as np


# Change the image size to 227x227. 
# Accuracy is much higher for squared images. 
# DO NOT MIX IT UP. 
img_width, img_height = 227, 227

train_data_dir = 'TrainingSets/'
nb_train_samples = 900 # number of samples in the training set

validation_data_dir = 'Validation/' 
nb_validation_samples = 150 # number of samples in the validation set

nb_epoch = 250 # how many epochs to train for. We are loading existing weights.
# so not needed unless training on new data

window_sz = 3 # how many pixels is the window that slides across the image is

# this will initiate a sequential backpropagation network
model = Sequential() 

# this adds 3 rows of zeros (black color pixels) to top of images and 3 columns
# to the sides. This is to prevent "washing away" of the sides. Convolutional nets
# tend to assume that anything on the edge is not important.
model.add(ZeroPadding2D(padding=(3, 3), input_shape=(227,227,3), data_format="channels_last"))

# 32 is the number of filters I first use. So it is the dimensionality of the output,
# or how many transformations the image goes through.
model.add(Conv2D(32, (window_sz, window_sz)))

#Batch Normalization helps the learning process to find consistent patterns in the batch
BatchNormalization()

# This adds a non-linear layer that is in our case Exponential Linear
# Unit. This is where learning happens through backpropagation.
model.add(ELU())

# Pooling layer, it is used to improve speed. Usually after we learned some things
# from initial image, it is harmless to downsample the image some. So we are pooling
# together every 4 pixels and taking an average, making it 1.
model.add(MaxPooling2D(pool_size=(2, 2)))

# The rest is just the above repeated five more times. As the network goes deeper, I include larger
# layers by increasing its filters
model.add(ZeroPadding2D(padding=(3, 3)))
model.add(Conv2D(32, (3, 3)))
BatchNormalization()
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(3, 3)))
model.add(Conv2D(64, (3, 3)))
BatchNormalization()
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(3, 3)))
model.add(Conv2D(64, (3, 3)))
BatchNormalization()
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(3, 3)))
model.add(Conv2D(128, (3, 3)))
BatchNormalization()
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(ZeroPadding2D(padding=(3, 3)))
model.add(Conv2D(256, (3, 3)))
BatchNormalization()
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))

#with each iteration we randomly turn off 20% of the neurons. This is a biological
#idea that works quite well. Basically, we are forcing the network to not focus
#too much on one single thing. If it does that, it becomes obsessed with little 
#patterns ignoring the big picture. So this is sort of like how brain reacts to 
#a sensory overload - receptors just start ignoring further stimulation.
model.add(Dropout(0.2))

#now we take the output which is a square and turn it into a 1D vector
model.add(Flatten())

#now that we have a vector we can put into a vector of 4096 Rectified Linear Units
#so so the final conclusion can be made
model.add(Dense(4096))
BatchNormalization()
model.add(Activation('elu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
#the last layer makes the decision. Decision is made by just 1 neuron, it says 
# fake or not.
model.add(Dense(1))
BatchNormalization()
model.add(Activation('sigmoid'))

#now the network is created.
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# this is the augmentation configuration we for training. This just creates
# additional images. So if we say choose to flip an image, we now have a normal image
# and a copy of it that is flipped.
train_datagen = ImageDataGenerator(
     rescale=1./255, #because neural nets like numbers in range 0-1, we divide by 255
     shear_range=0.3, #we shear the image a little
     zoom_range=0.3, #zoom in and out
     horizontal_flip=True, #randomly flip some images
     vertical_flip=True,
     samplewise_center=True,
     rotation_range=30,
     channel_shift_range=5)


# this is the augmentation configuration for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255,
                                  samplewise_center=True)

#History
history = History()

# checkpoint
#filepath="weightsS10a.best.h5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint, history]

#so to train the model I uncomment the followinf
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

# uncomment this to validate on the test set 
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

# this is where you train or fit the data, this line actually executes it.
model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        nb_epoch=nb_epoch,
        callbacks=callbacks_list,
        verbose=2)

        
