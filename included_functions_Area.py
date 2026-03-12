import numpy as np
import placentagen as pg
import matplotlib.image as mpimg
from skimage import filters, measure, color
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from skimage.morphology import skeletonize  #Compute the skeleton of a binary image
from skan import csr
from skan import Skeleton, summarize
from skan import draw
import cv2
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def read_png(filename, extract_colour):
    #This function reads in a png file and extract the relevant colour from the image
    img1 = mpimg.imread(filename)
    if extract_colour == 'all':
        img2 = img1
    elif extract_colour == 'r':
        img2 = img1[:, :, 0]
    elif extract_colour == 'g':
        img2 = img1[:, :, 1]
    elif extract_colour == 'b':
        img2 = img1[:, :, 2]
    else:  #default to all channels
        img2 = img1
    return img2

def get_scale(scalebar_size, image_array):

    #binary = img > 1.0e-6  #all non zeros

    line_pixels = np.where(image_array > 0.5)

    if len(line_pixels[0]) == 0:
        raise ValueError("No line detected in the image")

    # Extract the x-coordinates of the line
    x_coords = line_pixels[1]
    y_coords = line_pixels[0]
    # Calculate the length of the line in pixels
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)

    if x_range > y_range:  # Horizontal line
        length_in_pixels = x_range + 1
        print("Scale bar horizontal")
    else:  # Vertical line
        length_in_pixels = y_range + 1
        print("Scale bar vertical")

    print('Length of bar in pixels: ', length_in_pixels)


    # Calculate scale in mm/pixel
    scale_mm_per_pixel = scalebar_size / length_in_pixels
    scale_mm_per_pixel = np.round(scale_mm_per_pixel,4)

    return scale_mm_per_pixel


def calculate_area(image_path, scale, is_debug_image):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for green color in HSV
    lower_green = np.array([35, 50, 50])  # Adjust as needed
    upper_green = np.array([85, 255, 255])

    # Create a binary mask where green is detected
    mask = cv2.inRange(hsv, lower_green, upper_green)
    if is_debug_image:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Green Mask")
        plt.axis("off")

        plt.show()
    # Count the number of nonzero pixels (green area)
    green_pixels = np.count_nonzero(mask)

    # Convert pixels to real-world area (scale is in mm/pixel)
    area_mm2 = green_pixels * (scale ** 2)

    return green_pixels, area_mm2