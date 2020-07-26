import tensorflow as tf
from skimage.color import lab2rgb, rgb2lab
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,load_img)
import os
import numpy as np
from tqdm import tqdm



# Converts RGB valuse to LAB
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
    image_array=np.stack(images)       
    return image_array


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('./img_predictions/model.h5')


#Load test images
x = get_images("./OurTrainingImages/")
# print(len(test_images))

# make predictions
# for x in test_images:
output = model.predict(x)
output*=128
cur = np.zeros((256,256,3))
cur[:,:,0] = x[:,:,0] # L layer
cur[:,:,1:] = output[0] # A B layers
rgb_image = lab2rgb(cur)

img = array_to_img(rgb_image)
# img.save("./img_predictions/{}.jpg".format(i))
img.show() 