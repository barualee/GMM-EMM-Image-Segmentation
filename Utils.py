import numpy as np
from PIL import Image

HUES = [
    (255, 0, 0),   # red
    (0, 255, 0),  # green
    (0, 0, 255),   # blue
    (0, 255, 255), # cyan
    (255, 255, 0), # yellow
    (128, 0, 128), # purple
    (255, 165, 0), # orange
]

def image_load(file_name) :
    image = Image.open(file_name)
    image.load()
    result = np.asarray(image, dtype="int32")
    
    #checking image type as greyscale or color
    #0 for RGB, 1 for greyscale
    image_type = 1

    #check to see if image is greyscale, R=G=B channel pixel values.
    w, h = image.size
    for i in range(w):
        for j in range(h):
            r, g, b = image.getpixel((i,j))
            if r != g != b: 
                image_type = 0
    
    #convert to one channel if greyscale image
    if image_type == 1:
        result = result[:,:,0]
    return result
