#!/usr/bin/python
import PIL
from PIL import Image
import cv2
import numpy as np

# Read image
from src import ColourUtil

img = cv2.imread("../resources/img/Sw.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("../resources/img/Sw.jpg")
red_image = PIL.Image.open("../resources/img/Sw.jpg")

# setup parameters to detect a blob(tennis ball)
params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True

params.filterByColor = True
params.blobColor = 255;

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(img)

# empty list to store the keypoints with the systems colour and size requirements.
name = []


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob


# This loop goes through the keypoints and performs a check on the points RGB value.
for i in keypoints:
    x1 = i.pt[0]
    x2 = i.pt[1]
    rgb_pixel_value = ColourUtil.ColourCheck.get_pixels(red_image, x1, x2)

    if rgb_pixel_value[0] == 221 and rgb_pixel_value[1] == 251 and rgb_pixel_value[2] == 57:
        name.append(i)
        print(rgb_pixel_value)
        print(i)

# Keypoint with the right parameters and RGB value will be shown with a red circle around it.
img1 = cv2.drawKeypoints(img, name, np.array([]), (0, 0, 255),
                         cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

cv2.imshow("Keypoints", img1)
cv2.imshow("Keypoint", img3)
cv2.waitKey(0)
