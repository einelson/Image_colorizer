import numpy as np


images = np.ndarray(shape=(0,0,0,0)) #(image#, height, width, depth(channels))
image1 = np.zeros(shape=(156,156,1))
print(image1.shape)
# images=np.append(images, image1, axis=3)

print(images)
print(images.shape)