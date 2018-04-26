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


import os 
import numpy as np
img_width, img_height = 227, 227

train_data_dir = 'TrainingSets/'
nb_train_samples = 900 

validation_data_dir = 'Validation/' 
nb_validation_samples = 150 

window_sz = 5 # how many pixels is the window that slides across the image is

# this will initiate a sequential backpropagation network
model = Sequential() 
model.add(ZeroPadding2D(padding=(3, 3), input_shape=(227,227,3), data_format="channels_last"))
model.add(Conv2D(32, (window_sz, window_sz)))
BatchNormalization()
model.add(ELU())
model.add(MaxPooling2D(pool_size=(2, 2)))
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
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(2048))
BatchNormalization()
model.add(Activation('elu'))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(1))
BatchNormalization()
model.add(Activation('sigmoid'))

#what we do here is populate the models with the weights I previously learned
model.load_weights('weightsS3a.best.h5')

#now the network is created.
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1./255,
                                  samplewise_center=True)


#this tests your unlabeled files. you want to put appropriate directories here
files = os.listdir('Unlabeled/')

print('\ntest')
i = 0
with open("classification.txt", "w+") as of:#we save to this file
    for test_file in files: #the following is a little tricky, but I am using internal keras functions to manipulate the image
        if ".jpg" in test_file or ".JPG" in test_file:
            img = load_img(os.path.join('Unlabeled/', test_file))
            x = img_to_array(img) #we convert image to a vector
            x = test_datagen.standardize(x) #then we do the division over 255
            x = np.expand_dims(x, axis=0) #this is just so that the input vector is of the dimensions Keras likes.
            prediction = model.predict_classes(x, batch_size=1, verbose=0) #predict class label
            probability = model.predict_proba(x, batch_size=1, verbose=0) #predict probability - this is a bit finicky, neural nets are not probabilistic classifiers. Best they can do is estimate.
            if prediction[[0]] == 0:
                of.write(test_file + ',' + str(prediction[[0]]) + ',' + str(1.0 - probability[[0]]) + '\n')
            else:
                of.write(test_file + ',' + str(prediction[[0]]) + ',' + str(probability[[0]]) + '\n')
            #of.write(test_file + ',' + str(prediction[[0]]) + '\n')
            i += 1
            print(i)
        
