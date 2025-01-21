import cv2
import numpy as np
import gc
from PIL import Image, ImageChops


def resize_image(image, to_resize = None):
    if to_resize != None:
        h, w = to_resize
        if h==len(image) and w==len(image[0]):
            return image
        if h*w < image.shape[0]*image.shape[1]:
            interp = cv2.INTER_AREA
        elif h*w > image.shape[0]*image.shape[1]:
            interp = cv2.INTER_CUBIC
        else: 
            interp = cv2.INTER_LINEAR
        resized_image = cv2.resize(image, (h,w), interpolation=interp)
        return resized_image
    return image


def load_image(filepath, to_resize = None):
    image = cv2.imread(filepath)
    image = image[:,65:]   
    return resize_image(image, to_resize) 


def get_corners_from_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    non_zero_coords = cv2.findNonZero(gray_image)
    if non_zero_coords is None:
        raise ValueError("The input image has no non-black regions.")
    x, y, w, h = cv2.boundingRect(non_zero_coords)
    roi_coords = ((x, y), (x + w, y + h))
    return roi_coords


def seamless_merge(image1, image2, ref_image_contrib):
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
    image2_non_overlap[overlap_mask==255] = ref_image_contrib*image2[overlap_mask==255]
    image1_non_overlap[overlap_mask==255] = (1-ref_image_contrib)*image1[overlap_mask==255]

    # plt.imshow(overlap_mask)
    # plt.show()
    # Perform seamless cloning only in the non-overlapping regions
    # height, width, _ = image1.shape
    # center = (width // 2, height // 2)
    # merged_image = cv2.seamlessClone(image2_non_overlap, image1_non_overlap, np.zeros(image2.shape[0:2]).fill(255), center, cv2.MIXED_CLONE)
    merged_image = image1_non_overlap + image2_non_overlap
    # merged_image = cv2.seamlessClone(image2_overlap, merged_image, np.zeros(image2.shape[0:2]).fill(255), center, cv2.MIXED_CLONE, 1)

    del gray1, gray2, mask1, mask2, overlap_mask, image1_non_overlap, image2_non_overlap
    gc.collect()

    return merged_image


def selective_color_blur(image, target_color, kernel_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to select black pixels
    _, mask = cv2.threshold(gray_image, target_color, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_not(mask)
    # Apply blur only to pixels of the specified color
    blurred_image = np.copy(image)
    # blurred_image[mask == 255] = cv2.medianBlur(blurred_image, (kernel_size, kernel_size), 0)[mask == 255]
    blurred_image[mask == 255] = cv2.medianBlur(blurred_image, kernel_size)[mask == 255]
    del gray_image, mask
    gc.collect()
    return blurred_image

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
    corners = get_corners_from_image(im)
    im2 = get_roi_from_corners(im, corners[0], corners[1])
    return im2
    # im = Image.fromarray(im)
    # bg = Image.new(im.mode, im.size, (0,0,0))
    # # bg = Image.new(im.mode, im.size, (255,255,255))
    # diff = ImageChops.difference(im, bg)
    # diff = ImageChops.add(diff, diff, 2.0, -100)
    # bbox = diff.getbbox()
    # result = im
    # if bbox:
    #     result = im.crop(bbox)
    #     del im
    # del bbox, bg, diff
    # gc.collect()
    # return np.array(result)


def transform_keypoints_from_roi(keypoints_roi, roi_topleft):
    x,y = roi_topleft
    keypoints_image = list()
    for kp in keypoints_roi:
        # Offset the coordinates by the ROI's top-left corner
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        keypoints_image.append(kp)
    return keypoints_image


def get_roi_from_corners(image, p1, p2):
    x, y = p1
    w = p2[0] - p1[0]
    h = p2[1] - p1[1]
    roi_img = image[y:y + h, x:x + w]
    return roi_img