import cv2
import numpy as np
import gc
# from PIL import Image, ImageChops


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


def seamless_gradient_merge(warped_img1, img2, feather_pixels = 5):
    # Create base mask
    mask = np.zeros_like(warped_img1).astype(np.float64)
    mask[warped_img1 > 0] = 1

    # Create horizontal gradient
    for y in range(mask.shape[0]):
        non_zero = np.where(mask[y, :, 0] > 0)[0]
        if len(non_zero) > 0:
            left_edge = non_zero[0]
            right_edge = non_zero[-1]
            # Feather left edge
            for x in range(left_edge, min(left_edge + feather_pixels, mask.shape[1])):
                alpha = (x - left_edge) / feather_pixels
                mask[y, x] *= alpha
            # Feather right edge
            for x in range(max(0, right_edge - feather_pixels), right_edge + 1):
                alpha = (right_edge - x) / feather_pixels
                mask[y, x] *= alpha

    # Create vertical gradient
    for x in range(mask.shape[1]):
        non_zero = np.where(mask[:, x, 0] > 0)[0]
        if len(non_zero) > 0:
            top_edge = non_zero[0]
            bottom_edge = non_zero[-1]
            # Feather top edge
            for y in range(top_edge, min(top_edge + feather_pixels, mask.shape[0])):
                alpha = (y - top_edge) / feather_pixels
                mask[y, x] *= alpha
            # Feather bottom edge
            for y in range(max(0, bottom_edge - feather_pixels), bottom_edge + 1):
                alpha = (bottom_edge - y) / feather_pixels
                mask[y, x] *= alpha
    
    result = warped_img1 * mask + img2 * (1 - mask)
        
    return result.astype(np.uint8)


def seamless_merge_into_roi(image1, image2, roi_coords, ref_image_contrib):
    image2_roi = get_roi_from_corners(image2, roi_coords[0], roi_coords[1])
    image2_roi_limits = get_roi_from_corners(image2, roi_coords[0], roi_coords[1], img = False)
    # image2_p = image2.copy()
    image2[image2_roi_limits[0]:image2_roi_limits[1], image2_roi_limits[2]:image2_roi_limits[3]] = (image2_roi*ref_image_contrib) + ((1-ref_image_contrib)*image1)
    return image2


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


def pad_image(new_image_height, new_image_width, old_image_width, old_image_height, img, channels = 3):
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


def get_padded_images(img, ref):
    new_w = ref.shape[1] + (2*img.shape[1])
    new_h = ref.shape[0] + (2*img.shape[0])
    new_ref = pad_image(new_h, new_w, ref.shape[1], ref.shape[0], ref)
    new_img = pad_image(new_h, new_w, img.shape[1], img.shape[0], img)
    return new_img, new_ref


def crop_image(im):
    corners = get_corners_from_image(im)
    im2 = get_roi_from_corners(im, corners[0], corners[1])
    return im2


def transform_keypoints_from_roi(keypoints_roi, roi_topleft):
    x,y = roi_topleft
    keypoints_image = list()
    for kp in keypoints_roi:
        # Offset the coordinates by the ROI's top-left corner
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
        keypoints_image.append(kp)
    return keypoints_image


def get_roi_from_corners(image, p1, p2, img = True):
    x, y = p1
    w = p2[0] - p1[0]
    h = p2[1] - p1[1]
    roi_img = image[y:y + h, x:x + w]
    if img is True:
        return roi_img
    else:
        return (y, y + h, x, x + w)