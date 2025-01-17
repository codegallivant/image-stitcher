import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import copy
from PIL import Image, ImageChops
import glob
from ast import literal_eval
import time
import gc
import yaml

start_time = time.time()

modes = ["affine", "perspective"]

# Read the configuration file
with open("config.yaml", 'r') as ymlfile:
    cfg = yaml.safe_load(ymlfile)
    PROJECT_DIR = cfg["project_dir"]
    INPUT_DIR = cfg["input_dir"]
    STITCH_DIR = cfg["stitch_dir"]
    MIN_MATCH_COUNT = cfg["min_match_threshold"]
    RATIO = cfg["lowes_ratio"]
    MODE = modes[cfg["mode"]]
    CONSECUTIVE_RANGE = cfg["consecutive_range"]
    INPUT_LIMIT = cfg["input_limit"]


def load_image(filepath):
    image = cv2.imread(filepath)
    image = image[:,65:]    
    return image


class LazyList:
    def __init__(self, function, length):
        self.function = function
        self.length = length

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError("List index out of range")
            
        return self.function(index)
    
    def __len__(self):
        return self.length


dirs = [STITCH_DIR]

for dir in dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)


files = os.listdir(INPUT_DIR)
files = [os.path.join(INPUT_DIR, file) for file in files]
print(INPUT_DIR)

files = [file for file in files if file[-4:]==".png"]
if INPUT_LIMIT != None:
    files = files[:INPUT_LIMIT]
print(files)

files = [p[1] for p in enumerate(files) if p[0]%5==0]
def sort_key(s, a):
    p = os.path.basename(s)[:-4].split('_')
    p = [x.split('.') for x in p]
    pf = list()
    for xs in p:
        for x in xs:
            pf.append(x)
    return int(pf[a])
files = sorted(files, key=lambda x: (sort_key(x, 1), sort_key(x, 2), sort_key(x, 3)))

# files = files[:5]

#  read the images by taking their path
images = LazyList(lambda index: load_image(files[index]), len(files))
for i in range(0, len(files)):
    # print(files[i][-4:])
    # if files[i][-4:] == ".jpg":
    print("Displaying", files[i])
    # print(image)
    # image = image[:,65:]
    cv2.imshow('a', images[i])
    cv2.waitKey(1)
    # time.sleep(0.3)
    # images.append(images[i])

cv2.destroyAllWindows()

print(len(images), "images displayed")


sift = cv2.SIFT_create()
kp = list()
des = list()

emptyindexes = list()

for i in range(0, len(images)):
    # find the keypoints and descriptors with SIFT
    thiskp, thisdes = sift.detectAndCompute(images[i], None)
    if thisdes is None:
        emptyindexes.append(i)
    kp.append(thiskp)
    des.append(thisdes)


for index in sorted(emptyindexes, reverse=True):
    # del images[index]
    del kp[index]
    del des[index]
    del files[index]

gc.collect()


# BFMatcher with default params
bf = cv2.BFMatcher()



class DynamicConnectivity:
    def __init__(self, n):
        self.parent = [i for i in range(n)]
        self.size = [1 for _ in range(n)]

    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP == rootQ:
            return

        if self.size[rootP] < self.size[rootQ]:
            self.parent[rootP] = rootQ
            self.size[rootQ] += self.size[rootP]
        else:
            self.parent[rootQ] = rootP
            self.size[rootP] += self.size[rootQ]

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def get_connected_components(self):
        component_map = {}
        
        for i in range(len(images)):
            root = self.find(i)
            if root not in component_map:
                component_map[root] = list()
                component_map[root].append(i)
            else:
                component_map[root].append(i)
        components = list(component_map.values())
        return components
    
connections = DynamicConnectivity(len(images))


# Determining which image is connected with which image
matchedimages = list()

# plt.figure()

# f, axarr = plt.subplots(len(images), len(images))

def deep_copy_dmatches(dmatches):
            return [[cv2.DMatch(_d[0].queryIdx, _d[0].trainIdx, _d[0].imgIdx, _d[0].distance)] for _d in dmatches]
        
for i in range(0, len(images)):
    # cv.drawMatchesKnn expects list of lists as matches.
    matchedimages.append(list())
    for j in range(0, len(images)):

        # Remove if related images not consecutive
        if j==i:
            continue
        if CONSECUTIVE_RANGE != None:
            if abs(j-i)>CONSECUTIVE_RANGE:
                continue
        
        
        matches = bf.knnMatch(des[i],des[j],k=2)
        # Apply ratio test
        good = list()
        for match in matches:
            if len(match)>1:
                if match[0].distance < RATIO*match[1].distance:
                    good.append([match[0]])
        good2 = deep_copy_dmatches(good)
                
        matchedimage = cv2.drawMatchesKnn(images[i],kp[i],images[j],kp[j],good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        matchedimages[i].append(matchedimage)
        # axarr[i][j].imshow(matchedimages[i][j])
        # good = []
        # for x in matches:
        #     if len(x) > 1:
        #         if x[0].distance < 0.7*x[1].distance:
        #             good.append(x[0])
        del good
        good = good2
        if len(good)>MIN_MATCH_COUNT:
            print( "{}->{}, Matches found - {}/{}".format(i,j,len(good), MIN_MATCH_COUNT) )
            connections.union(i, j)
        else:
            print( "{}->{}, Not enough matches found - {}/{}".format(i,j,len(good), MIN_MATCH_COUNT) )
    
    collections = connections.get_connected_components()
    collections = sorted(collections, key=len, reverse=True)
    print(collections)

# plt.show()

# collections = connections.get_connected_components()
# collections = sorted(collections, key=len, reverse=True)
# print(collections)


def seamless_merge(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Find non-zero regions in both images
    _, mask1 = cv2.threshold(gray1, 10, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(gray2, 10, 255, cv2.THRESH_BINARY)

    # Invert the masks
    # mask1_inv = cv2.bitwise_not(mask1)
    # mask2_inv = cv2.bitwise_not(mask2)

    # Combine the inverted masks to get the region of overlap
    overlap_mask = cv2.bitwise_and(mask1, mask2)
    # overlap_mask = cv2.bitwise_not(overlap_mask)
    image2_non_overlap = np.copy(image2)
    image1_non_overlap = np.copy(image1)
    image2_non_overlap[overlap_mask==255] = 0.6*image2[overlap_mask==255]
    image1_non_overlap[overlap_mask==255] = 0.4*image1[overlap_mask==255]

    # plt.imshow(overlap_mask)
    # plt.show()
    # Perform seamless cloning only in the non-overlapping regions
    height, width, _ = image1.shape
    center = (width // 2, height // 2)
    # merged_image = cv2.seamlessClone(image2_non_overlap, image1_non_overlap, np.zeros(image2.shape[0:2]).fill(255), center, cv2.MIXED_CLONE)
    merged_image = image1_non_overlap + image2_non_overlap
    # merged_image = cv2.seamlessClone(image2_overlap, merged_image, np.zeros(image2.shape[0:2]).fill(255), center, cv2.MIXED_CLONE, 1)

    del gray1, gray2, mask1, mask2, overlap_mask, image1_non_overlap, image2_non_overlap
    gc.collect()

    return merged_image


def selective_color_blur(image, target_color, color_threshold, kernel_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to select black pixels
    _, mask = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    # Apply blur only to pixels of the specified color
    blurred_image = np.copy(image)
    # blurred_image[mask == 255] = cv2.medianBlur(blurred_image, (kernel_size, kernel_size), 0)[mask == 255]
    blurred_image[mask == 255] = cv2.medianBlur(blurred_image, kernel_size)[mask == 255]
    del gray_image, mask
    gc.collect()
    return blurred_image


# create new image of desired size and color for padding
def getpaddedimg(new_image_height, new_image_width, old_image_width, old_image_height, img, channels = 3):
    color = (0,0,0)
    if channels == 1:
        result = np.full((new_image_height,new_image_width), color[0:1], dtype=np.uint8)
    else:
        result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img
    return result


def crop_image(im):
    im = Image.fromarray(im)
    bg = Image.new(im.mode, im.size, (0,0,0))
    # bg = Image.new(im.mode, im.size, (255,255,255))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    result = im
    if bbox:
        result = im.crop(bbox)
        del im
    del bbox, bg, diff
    gc.collect()
    return np.array(result)


# stitch_order = list()
# image_warps = list()

image_shapes = [image.shape for image in images]

unblended_collections = copy.deepcopy(collections)
blended_collections = list()

i = 0
for unblended_image_group in unblended_collections:
    unblended_image_indexes = copy.deepcopy(unblended_image_group)
    ref = unblended_image_indexes[0]
    reference_image = images[ref]
    unblended_image_indexes.remove(ref)

    # stitch_order.append(list())
    # image_warps.append(list())
    # stitch_order[i].append(ref)
    # image_warps[i].append(reference_image)
    
    key_points_broken = False
    
    while len(unblended_image_indexes)!=0:
        print(unblended_image_indexes)
        matched_once = False
        for k in unblended_image_indexes:
            new_image_width = reference_image.shape[1] + (2*image_shapes[k][1])
            new_image_height = reference_image.shape[0] + (2*image_shapes[k][0])
            reference_image = getpaddedimg(new_image_height, new_image_width, reference_image.shape[1], reference_image.shape[0], reference_image)
            current_image = getpaddedimg(new_image_height, new_image_width, image_shapes[k][1], image_shapes[k][0], images[k])

            thiskp, thisdes = sift.detectAndCompute(current_image, None)
            refkp, refdes = sift.detectAndCompute(reference_image, None)

            if refdes is None:
                print(f"{i} {k} Key points not found in ref. Moving to next blend.")
                unblended_collections.append(unblended_image_indexes[k:])
                key_points_broken = True
                break
                
            
            
            # print(thisdes.dtype, refdes.dtype)
            matches = bf.knnMatch(thisdes,refdes,k=2)
            good = list()
            for match in matches:
                if match[0].distance < RATIO*match[1].distance:
                    good.append([match[0]])
            good2 = deep_copy_dmatches(good)
            matchedimage = cv2.drawMatchesKnn(current_image, thiskp, reference_image,refkp,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            del good
            good = good2
            if len(good)>=MIN_MATCH_COUNT:
                matched_once = True
                src_pts = np.float32([thiskp[m[0].queryIdx].pt for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([refkp[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)
                # print(src_pts)
                # print(dst_pts)
                if MODE == "affine":
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    warped_image = cv2.warpAffine(current_image, M, (reference_image.shape[1], reference_image.shape[0]))
                    # if FIND_ROUTE is True:
                    #     for tpi in range(0, len(track_points)):
                    #         # Take into account padding and transform
                    #         if k == point_img_index[tpi]:
                    #             # Going to be blended
                    #             homogenous_coords = np.array([track_points[tpi][0]+((new_image_height-images[k].shape[0])/2), track_points[tpi][1]+((new_image_width-images[k].shape[1])/2), 1])
                    #         # if (point_img_index[tpi] in unblended_image_group and point_img_index[tpi] not in unblended_image_indexes):
                    #         #     # Already blended
                    #         #     homogenous_coords = np.array([track_points[tpi][0]+((new_image_height-reference_image.shape[0])/2), track_points[tpi][1]+((new_image_width-reference_image.shape[1])/2), 1])
                    #         if k == point_img_index[tpi]:
                    #             track_points[tpi] = np.matmul(M, homogenous_coords)
                    #             print("tp",track_points)
                elif MODE == "perspective":
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                    warped_image = cv2.warpPerspective(current_image, M, (reference_image.shape[1], reference_image.shape[0]))

                # stitch_order[i].append(k)
                # image_warps[i].append(warped_image) # This image warp will be used for prediction
                # alpha = 1.0
                # beta = 1.0
                # reference_image = cv2.addWeighted(stitched_image, alpha, reference_image, beta, 0.0)
                # reference_image = cv2.bitwise_or(reference_image, warped_image)
                # reference_image = cv2.seamlessClone(warped_image, reference_image, np.zeros(reference_image.shape), (0,0), cv2.MIXED_CLONE)
                reference_image = seamless_merge(warped_image, reference_image)
                # reference_image = cv2.seamlessClone(warped_image, reference_image, mask, (reference_image.shape[0]//2, reference_image.shape[1]//2), cv2.MIXED_CLONE)
                # reference_image = alpha_blend_with_mask(reference_image, warped_image, 1)

                # cv2.imshow(f"stitch {k}",reference_image)
                # while True:
                #     key = cv2.waitKey(0) & 0xFF
                #     print(k)
                #     if key == ord('c'): # you can put any key here
                #         cv2.destroyAllWindows()
                #         break
                
                unblended_image_indexes.remove(k) 
            else:
                print( "{}->ref, Not enough matches are found - {}/{}".format(k,len(good), MIN_MATCH_COUNT) )

            # if FIND_ROUTE is True:
            #     for tpi in range(0, len(track_points)):
            #         if k == point_img_index[tpi] or (point_img_index[tpi] in unblended_image_group and point_img_index[tpi] not in unblended_image_indexes):
            #             # Take into account cropping
            #             print(track_points)
            #             print(get_black_pixels_before_color(reference_image))
            #             track_points[tpi] = np.subtract(np.array(track_points[tpi]),np.array([get_black_pixels_before_color(reference_image,"rows"), get_black_pixels_before_color(reference_image,"columns")]))
            #             print("tp",track_points)

            reference_image = crop_image(reference_image) # Removing padding
            
            
        if matched_once == False:
            # reference_image = crop_image(reference_image)
            break

        if key_points_broken == True:
            break
    
        

    # reference_image = cv2.medianBlur(reference_image, 3)     # Removing pepper noise developed during bitwise OR
    blended_collections.append(reference_image)
    i = i+1


print("Number of blends formed:", len(blended_collections))

delfiles = glob.glob(f'{STITCH_DIR}/*')
for f in delfiles:
    os.remove(f)

i = 0

for i in range(0, len(blended_collections)):
    blended_collections[i] = selective_color_blur(blended_collections[i], (0,0,0), 30, 17)
#     blended_collections[i] = cv2.medianBlur(blended_collections[i], 5)     # Removing pepper noise developed during bitwise OR



for blend in blended_collections:
    stitchpath = os.path.join(STITCH_DIR,f"stitched{i}.jpg")
    print(stitchpath)
    cv2.imwrite(stitchpath, blend)
    # cv2.imwrite(stitchpath, selective_color_blur(blend, (0,0,0), 20, 9))
    i = i + 1
    # while True:
    #     key = cv2.waitKey(0) & 0xFF
    #     print(k)
    #     if key == ord('c'): # you can put any key here
    #         cv2.destroyAllWindows()
    #         break

end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")