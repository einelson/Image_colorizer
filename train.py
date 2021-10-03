'''
train.py
Overview:
    This file will be for gathering training data and for training algorithm

'''



# image handling
import os
from numpy.core.defchararray import index
from tqdm import tqdm
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import (img_to_array,load_img)
from cv2 import cv2

# neural network
import tensorflow as tf


'''
load_images
loads images as array and saves them in an array
'''
# Converts RGB valuse to LAB
def get_lab(img):
    l = rgb2lab(img/255)[:,:,0]
    return l

# Returns the AB values from the LAB
def get_color(img):
    x = rgb2lab(img/255)[:,:,1:] # this is the A and B values; a-magenta-green; b-yellow-blue
    x = x/128
    return x

def load_images(path='', color=False):
    images = list()
    print('Loading Images')
    for filename in tqdm(os.listdir(path)):
        if filename[0] != '.':
            if color == False:
                img = get_lab(np.array(img_to_array(load_img(path + filename)), dtype=float))
                images.append(img.reshape(img.shape[0],img.shape[1],1))
            else:
                img = get_color(np.array(img_to_array(load_img(path + filename)), dtype=float))
                images.append(img.reshape(img.shape[0],img.shape[1],2))
    image_array=np.stack(images)            
    return image_array

'''
train
Opens images and trains the neural network
'''
def train():
    # open images
    l = load_images(path='train_images/', color=False) # only l
    ab = load_images(path='train_images/', color=True) # a and b values


    print('train shape: {}'.format(l.shape))
    print('test shape: {}'.format(ab.shape))

    # split between test and training data
    X_train, X_test, Y_train, Y_test = train_test_split(l, ab, test_size=0.15)
    
    # create model
    inputs=tf.keras.Input(shape=(256, 256, 1))

    # block 1 -- input black and white image
    x=tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)(inputs)
    x=tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(x)
    x=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)    
    x=tf.keras.layers.UpSampling2D((2, 2))(x)
    x=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    x=tf.keras.layers.UpSampling2D((2, 2))(x)
    x=tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x=tf.keras.layers.UpSampling2D((2, 2))(x)
    x=tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x=tf.keras.layers.UpSampling2D((2, 2))(x)
    outputs=tf.keras.layers.Conv2D(2, (3, 3), activation='tanh', padding='same')(x)
    # end model

    # plot model
    model=tf.keras.Model(inputs=inputs, outputs=outputs, name="image_colorization")
    # tf.keras.utils.plot_model(model, "./saved models/model.png", show_shapes=True)

    # model summary
    # model.summary()

    # compile model and fit with training data
    model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x=X_train, y=Y_train,epochs=10000,batch_size=10, validation_data=(X_test,Y_test))


    # save model
    # print(os.getcwd()+'/saved models/model.h5')
    model.save(os.getcwd()+'/saved models/model.h5') 


    # model accuracy
    _, acc = model.evaluate(x=X_test, y=Y_test)
    acc = 100*(acc)
    if acc > 90:
        print('Accuracy: {}%'.format(acc))
    elif acc < 89:
        print('Accuracy is very low: {}%'.format(acc))
    elif acc < 10:
        print('Accuracy is very low. Retraining is necessary to have a working model: {}%'.format(acc))

    # predict one image
    # first is channel (image number)
    xt = l[0] 
    test_image= np.stack([xt, xt])  
    print('testing stack shape: {}'.format(test_image[0].shape))

    # test network on an image
    colored = (model.predict(test_image)[0]) *128
    print('predicted shape: {}'.format(colored.shape))

    # colored = add l value again
    cur = np.zeros((256,256,3))
    cur[:,:,0] = xt[:,:,0] # L layer
    cur[:,:,1:] = colored # A B layers
    rgb_image = lab2rgb(cur) 
    print('final shape: {}'.format(rgb_image.shape))
    cv2.imshow('test image', rgb_image)
    cv2.waitKey(0)

# run
if __name__ == "__main__":
    train()