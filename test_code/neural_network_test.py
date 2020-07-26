# import libraries
import tensorflow as backend
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, InputLayer, UpSampling2D

import pickle

'''
Convert all training images from the RGB color space to the Lab color space.
Use the L channel as the input to the network and train the network to predict the ab channels.
Combine the input L channel with the predicted ab channels.
Convert the Lab image back to RGB.
'''

# import train and target data
# colored photo to train on
train0 = np.array(img_to_array(load_img("./Image_Colorizer/TestImages/0AEYvu.jpg")), dtype=float)
# print(train0.shape)---(256, 256, 3)

# convert data from RGB to LAB and normalize-(we normalize by diving by 255--this gives us a value between 0 and 1)
x_train = rgb2lab(train0/255)[:,:,0] # this is the L layer- the black and white values
x_target = rgb2lab(train0/255)[:,:,1:] # this is the A and B values; a-magenta-green; b-yellow-blue
x_target/=128

# The Conv2D layer we will use later expects the inputs and training outputs to be of the following format:
# (samples, rows, cols, channels), so we need to do some reshaping
# https://keras.io/layers/convolutional/
x = x_train.reshape(1, x_train.shape[0], x_train.shape[1], 1)
y = x_target.reshape(1, x_target.shape[0], x_target.shape[1], 2)

# create model
model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1))) # input shape is only needed for first layer? input_shape=(256, 256, 3)
# 3x3 kernel used and 8 filters?
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
# figure out what this does
# model.add(layers.MaxPooling2D((2, 2)))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3,3), activation='tanh', padding='same'))
# get summary of layers and compile
model.summary()
model.compile(optimizer='adam',loss='mse') # loss='sparse_categorical_crossentropy', optomizer='rmsprop'
model.fit(x=x,y=y, batch_size=50,verbose=1, epochs=500)
# evaluate model
model.evaluate(x, y, batch_size=1)

# our black and white photo to test on
train0 = np.array(img_to_array(load_img("./Image_Colorizer/TestImages/swim.jpg")), dtype=float)
x_train = rgb2lab(train0/255)[:,:,0] # this is the L layer- the black and white values
x_target = rgb2lab(train0/255)[:,:,1:] # this is the A and B values; a-magenta-green; b-yellow-blue
x_target/=128
x = x_train.reshape(1, x_train.shape[0], x_train.shape[1], 1)
y = x_target.reshape(1, x_target.shape[0], x_target.shape[1], 2)

# make predictions
output = model.predict(x)
print(output.shape)
output*=128

# make sure output has the correct shape
cur = np.zeros(train0.shape)
cur[:,:,0] = x[0][:,:,0] # L layer?
cur[:,:,1:] = output[0] # A B layers?
print(cur.shape)

# convert to rgb
rgb_image = lab2rgb(cur)

img = array_to_img(rgb_image)
# img.save("./Image_Colorizer/img_predictions/001.jpg")
img.show()