import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# var=np.array([[1,2,3], [1,2,3]])

# print(var.shape)
# print(var)
storage=list()

img1=np.array(img_to_array(load_img("./TestImages/0AEYvu.jpg")), dtype=float)
# print(img1.shape)
storage.append(img1)

img2=np.array(img_to_array(load_img("./TestImages/0AEYvu.jpg")), dtype=float)
storage.append(img2)

img3=np.array(img_to_array(load_img("./TestImages/0AEYvu.jpg")), dtype=float)
storage.append(img3)


print('expected')
imgs = np.stack((storage), axis=3)
print(imgs.shape)
# print all info from 4th dimension
# print(imgs[:,:,:,1])
