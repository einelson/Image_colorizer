import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,load_img)
import os
import numpy as np
from tqdm import tqdm
import cv2 as cv2



# Converts RGB values to LAB
def get_lab(img):
    l = rgb2lab(img/255)[:,:,0]
    return l

# Gets the images and returns them in a 4d np.array format
def get_images(path):
    images = list()
    print('Loading Images')
    for filename in tqdm(os.listdir(path)):
        if filename[0] != '.':
            img = get_lab(np.array(img_to_array(load_img(path + filename)), dtype=float))
            images.append(img.reshape(img.shape[0],img.shape[1],1)) 
    # image_array=np.stack(images)       
    return images


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('./saved models/model.h5')


#Load test images
images = get_images("./train_images/")
# print(len(test_images))

# make predictions
# for x in test_images:
for x in images:
    print(x.shape)
    test_image= np.stack([x, x])  
    print('testing stack shape: {}'.format(test_image[0].shape))

    # test network on an image
    output = (model.predict(test_image)[0]) *128
    print(output.shape)
    cur = np.zeros((256,256,3))
    cur[:,:,0] = x[:,:,0] # L layer
    cur[:,:,1:] = output # A B layers
    rgb_image = lab2rgb(cur)

    cv2.imshow('test image', rgb_image)
    cv2.waitKey(0)