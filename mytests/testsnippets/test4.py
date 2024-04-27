import cv2
import os
import numpy as np

colour = (110, 193, 228)
maskdest = "/mnt/d/datasets/semantic-segmentation-of-aerial-images/dataset/masks"
for mask in os.listdir(maskdest):
    maskpath = os.path.join(maskdest, mask)
    im = cv2.imread(maskpath)
    im[np.all(im != colour, axis=-1)] = (0,0,0) 
    im[np.all(im != (0,0,0), axis=-1)] = (255,255,255)
    cv2.imwrite(maskpath, im)