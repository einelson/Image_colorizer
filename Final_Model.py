# import libraries
import os

import numpy as np
import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, InputLayer, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,load_img)
from tqdm import tqdm


def get_lab(img):
    l = rgb2lab(img/255)[:,:,0]
    return l

def get_color(img):
    x = rgb2lab(img/255)[:,:,1:] # this is the A and B values; a-magenta-green; b-yellow-blue
    x/=128
    return x

def get_images(path, color="lab"):
    images = np.array()
    for filename in os.listdir(path):
        if filename[0] != '.':
            if color == "lab":
                img = get_lab(np.array(img_to_array(load_img(path + filename)), dtype=float))
                images.append(img.reshape(1,img.shape[0],img.shape[1],1))
            else:
                img = get_color(np.array(img_to_array(load_img(path + filename)), dtype=float))
                images.append(img.reshape(1,img.shape[0],img.shape[1],2))
    return images

'''
Convert all training images from the RGB color space to the Lab color space.
Use the L channel as the input to the network and train the network to predict the ab channels.
Combine the input L channel with the predicted ab channels.
Convert the Lab image back to RGB.
'''
x = get_images("./OurTrainingImages/") #l value only
print(len(x))
y = get_images("./OurTrainingImages/", color="yes") #a and b values

print(x.shape())

# Recreate the exact same model, including its weights and the optimizer
# model = tf.keras.models.load_model('./img_predictions/model.h5')

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
# get working after we get NN working better
'''
# supposed to soften image
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
'''
# get summary of layers and compile
model.summary()
model.compile(optimizer='adam',loss='mse') # loss='sparse_categorical_crossentropy', optomizer='rmsprop'


# there is an issue fitting the data
for e in tqdm(range(10000)):
    for i,j in enumerate(x):
        model.fit(x=x[i],y=y[i], batch_size=50,verbose=0, epochs=1)

# evaluate model
# model.evaluate(x, y, batch_size=1)

# save model
model.save('./img_predictions/model.h5') 


#Load test images
test_images = get_images("./OurTrainingImages/")
# print(len(test_images))

for i,z in enumerate(test_images):
    # make predictions
    output = model.predict(z)
    output*=128
    cur = np.zeros((256,256,3))
    cur[:,:,0] = z[:,:,0] # L layer?
    cur[:,:,1:] = output[0] # A B layers?
    rgb_image = lab2rgb(cur)

    img = array_to_img(rgb_image)
    img.save("./img_predictions/{}.jpg".format(i))
    img.show() 
