#!/usr/bin/python

# Standard imports
import cv2
import numpy as np

# Read image
img = cv2.imread("img/Sw.jpg", cv2.IMREAD_GRAYSCALE)
# sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
#
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)

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

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
for i in keypoints :
    print(i.pt[0])
    print(i.pt[1])
    print(i.response)

print(keypoints)

img1 = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
# sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
# sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)
#
# sobelX = np.uint8(np.absolute(sobelX))
# sobelY = np.uint8(np.absolute(sobelY))
#
# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
cv2.imshow("Keypoints", img1)
cv2.waitKey(0)