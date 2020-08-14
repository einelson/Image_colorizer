# import libraries
import os

import numpy as np
import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, InputLayer, UpSampling2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,load_img)
from tqdm import tqdm

# Converts RGB valuse to LAB
def get_lab(img):
    l = rgb2lab(img/255)[:,:,0]
    return l

# Returns the AB values from the LAB
def get_color(img):
    x = rgb2lab(img/255)[:,:,1:] # this is the A and B values; a-magenta-green; b-yellow-blue
    x/=128
    return x

# Gets the images and returns them in a 4d np.array format
def get_images(path, color="l"):
    images = list()
    print('Loading Images')
    for filename in tqdm(os.listdir(path)):
        if filename[0] != '.':
            if color == "l":
                img = get_lab(np.array(img_to_array(load_img(path + filename)), dtype=float))
                images.append(img.reshape(img.shape[0],img.shape[1],1))
            else:
                img = get_color(np.array(img_to_array(load_img(path + filename)), dtype=float))
                images.append(img.reshape(img.shape[0],img.shape[1],2))
    image_array=np.stack(images)            
    return image_array

'''
Convert all training images from the RGB color space to the Lab color space.
Use the L channel as the input to the network and train the network to predict the ab channels.
Combine the input L channel with the predicted ab channels.
Convert the Lab image back to RGB.
'''
# Get the LAB values
x = get_images("./OurTrainingImages/") #l value only
y = get_images("./OurTrainingImages/", color="yes") #a and b values

# Check the array shape (256,256,1,x(number of images))
print(x.shape)
print(y.shape)

# create model needs to be remade for a 4d array...
model = Sequential()
model.add(InputLayer(input_shape=(256, 256, 1))) # input shape is only needed for first layer? input_shape=(256, 256, 1)
# 3x3 kernel used and 8 filters
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.summary()
# Finish model
model.compile(optimizer='rmsprop',loss='mse')

# there is an issue fitting the data
model.fit(x=x,y=y, batch_size=1,verbose=1, epochs=1000)

# evaluate model
# model.evaluate(x, y, batch_size=1)

# save model
model.save('./img_predictions/model.h5') 


# #Load test images
# test_images = get_images("./OurTrainingImages/")
# # print(len(test_images))

# # make predictions
# for x in test_images:
#     output = model.predict(x)
#     output*=128
#     cur = np.zeros((256,256,3))
#     cur[:,:,0] = x[:,:,0] # L layer
#     cur[:,:,1:] = output[0] # A B layers
#     rgb_image = lab2rgb(cur)

#     img = array_to_img(rgb_image)
#     # img.save("./img_predictions/{}.jpg".format(i))
#     img.show() 