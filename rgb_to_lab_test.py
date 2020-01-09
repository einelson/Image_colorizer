# import library
from keras.preprocessing.image import array_to_img
from skimage import io, color


# open image
print("Opening image...")
rgb = io.imread("C:/Users/einel/OneDrive/Pictures/Wallpapers/1140046.jpg")

# convert image to LAB values (this is an array)
lab = color.rgb2lab(rgb)
print("lab colors...")
print(lab)

# revert to rgb values
back_to_rgb = color.lab2rgb(lab)
print("rgb colors...")
print(back_to_rgb)

# show image // needs help
print("image")
img_pil = array_to_img(back_to_rgb)
img_pil.show()
