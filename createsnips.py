import os
import random
import cv2
import numpy as np
import random
import glob
from PIL import Image, ImageChops

# Path to the directory containing images
images_dir = "/mnt/d/manas-taskphase/mapthemap/imagedata"

# Output directory to save cropped images
output_dir = "/mnt/d/manas-taskphase/mapthemap/image_fragments8"


# Number of random snippets to generate from each image
num_snippets_per_image = 8

# Size of the snippets (width, height)
snippet_size = (random.randint(800,800), random.randint(800,800))

# Maximum rotation angle in degrees
# max_rotation_angle = 0

def crop_image(im):
    im = Image.fromarray(im)
    bg = Image.new(im.mode, im.size, (0,0,0))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    result = im
    if bbox:
        result = im.crop(bbox)
    return np.array(result)

def main():
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    delfs = glob.glob(os.path.join(output_dir, '*'))
    for f in delfs:
        os.remove(f)

    # List all image files in the input directory
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif'))]
    # image_files = ["map1.jpg"]

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(images_dir, image_file)
        img = cv2.imread(image_path)

        # Get image dimensions
        height, width, _ = img.shape

        # Generate random snippets from the image
        for i in range(num_snippets_per_image):
            # Randomly choose the top-left corner of the snippet
            x = random.randint(0, width - snippet_size[0])
            y = random.randint(0, height - snippet_size[1])

            # Crop the snippet from the image
            snippet = img[y:y+snippet_size[1], x:x+snippet_size[0]]

            # Randomly rotate the snippet
            # angle = random.uniform(-max_rotation_angle, max_rotation_angle)
            angle = random.choice([0,90,180])
            rotated_snippet = rotate_image(snippet, angle)
            rotated_snippet = crop_image(rotated_snippet)

            # Save the rotated snippet to the output directory
            snippet_filename = f"{os.path.splitext(image_file)[0]}_{i}.png"
            snippet_output_path = os.path.join(output_dir, snippet_filename)
            cv2.imwrite(snippet_output_path, rotated_snippet)

            print(f"Saved snippet: {snippet_output_path}")

def rotate_image(image, angle):
    """Rotate an image by a given angle."""
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    return rotated_image

if __name__ == "__main__":
    main()
