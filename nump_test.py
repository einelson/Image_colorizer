import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# var=np.array([[1,2,3], [1,2,3]])

# print(var.shape)
# print(var)

img1=np.array(img_to_array(load_img("./TestImages/0AEYvu.jpg")), dtype=float)
# print(img1.shape)

img2=np.array(img_to_array(load_img("./TestImages/0AEYvu.jpg")), dtype=float)

img3=np.array(img_to_array(load_img("./TestImages/0AEYvu.jpg")), dtype=float)

imgs = np.stack((img1,img2,img3), axis=3)
print(imgs.shape)
# print all info from 4th dimension
print(imgs[:,:,:,1])