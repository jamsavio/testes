import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def unsharp_mask(image, kernel_size=(5, 5), sigma=2.0, amount=2.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

img = cv.imread('caminho da img')

#filter
dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)

#aplicar nitidez
sharpened_image = unsharp_mask(dst)

#aumentar contraste
lookUpTable = np.empty((1,256), np.uint8)
for i in range(256):
	lookUpTable[0,i] = np.clip(pow(i / 255.0, 0.6) * 255.0, 0, 255)
res = cv.LUT(sharpened_image, lookUpTable)

cv.imshow('img', res) # Display img with median filter
#cv.imwrite('output2.jpg', res)
cv.waitKey(0)        # Wait for a key press to
cv.destroyAllWindows # close the img window.
