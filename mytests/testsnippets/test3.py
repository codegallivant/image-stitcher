import os
import shutil
# Preprocessing aerial-semantic-segmentation-dataset

folderpath = "/mnt/d/datasets/semantic-segmentation-of-aerial-images/archive/Semantic segmentation dataset"

imagedest = "/mnt/d/datasets/semantic-segmentation-of-aerial-images/dataset/images"
maskdest = "/mnt/d/datasets/semantic-segmentation-of-aerial-images/dataset/masks"

i = 0
for subfolder in os.listdir(folderpath):
    print(subfolder)
    subfolderpath = os.path.join(folderpath, subfolder)
    if os.path.isdir(subfolderpath):
        imagefolderpath = os.path.join(subfolderpath, "images")
        masksfolderpath = os.path.join(subfolderpath, "masks")
        for image in os.listdir(os.path.join(subfolderpath, "images")):
            shutil.copy(os.path.join(imagefolderpath, image), os.path.join(imagedest, "image"+str(i)+".jpg"))
            thismaskpath = os.path.join(masksfolderpath, os.path.splitext(image)[0]+".png")
            if os.path.isfile(thismaskpath):
                shutil.copy(thismaskpath, os.path.join(maskdest, "image"+str(i)+".jpg"))
            i = i+1
        print("Copied images and masks from", subfolder)
