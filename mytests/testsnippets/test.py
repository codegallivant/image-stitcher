import cv2
import numpy as np
import sys
# Load the images
image1 = cv2.imread('map1snip1.jpg')
image2 = cv2.imread('map1snip2.jpg')


# compute center offset
def getpaddedimg(new_image_height, new_image_width, old_image_width, old_image_height, img, channels = 3):
    color = (0,0,0)
    result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img
    return result
# create new image of desired size and color (blue) for padding
new_image_width = image1.shape[1] + image2.shape[1]
new_image_height = image1.shape[0] + image2.shape[0]
image1 = getpaddedimg(new_image_height, new_image_width, image1.shape[1], image1.shape[0], image1)
image2 = getpaddedimg(new_image_height, new_image_width, image2.shape[1], image2.shape[0], image2)

# # Convert images to grayscale
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)

# Initialize brute force matcher
bf = cv2.BFMatcher()

# Perform brute force matching
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test to obtain good matches
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extract keypoints corresponding to good matches
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Use RANSAC to estimate homography
homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)





# Warp image1 onto image2 using the estimated homography
stitched_image = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

alpha = 1.0
beta = 1.0
dst = cv2.addWeighted(stitched_image, alpha, image2, beta, 0.0)
# Display the stitched image
cv2.imshow('image1', image1)
cv2.imshow('image2', image2)
cv2.imshow('Stitched Image', stitched_image)
cv2.imshow('dst', dst)

while True:
    k = cv2.waitKey(0) & 0xFF
    print(k)
    if k == ord('c'): # you can put any key here
        cv2.destroyAllWindows()
        break

