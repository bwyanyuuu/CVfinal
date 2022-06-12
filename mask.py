import numpy as np
import cv2

img = cv2.imread('./output/f1.png')

mask = img == 0
one = np.ones(mask.shape)
mask = mask*one

cv2.imwrite('./output/f1_mask.png', mask*255)