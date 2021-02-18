import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import imutils

img = cv.imread("img/courtG.jpg")

greenLower = np.array([98, 111, 98])
greenUpper = np.array([120, 181, 152])

gaussBlur = cv.GaussianBlur(img, (11, 11), 0)
hsv = cv.cvtColor(gaussBlur, cv.COLOR_BGR2HSV)
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask1 = cv.inRange(hsv, greenLower, greenUpper)
mask2 = cv.erode(mask1, None, iterations=2)
mask3 = cv.dilate(mask2, None, iterations=2)
_, mask = cv.threshold(mask3, 220, 255, cv.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

# dilation = cv.dilate(mask, kernal, iterations = 3)
erosion = cv.erode(mask, kernal, iterations=3)
# opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernal)
# closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernal)
# mg = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernal)
# th = cv.morphologyEx(mask, cv.MORPH_TOPHAT, kernal)
#
# titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
# images = [img, mask, dilation, erosion, opening, closing, mg, th]

# for i in range(8):
#     plt.subplot(2, 4, i+1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# _,th2 = cv.threshold(erosion, 200, 255, cv.THRESH_BINARY_INV)
ret, img1 = cv.threshold(erosion, 125, 255, cv.THRESH_BINARY_INV)


contours, hierachy = cv.findContours(img1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

threshold_blobs_area = 5500

for i in range(1, len(contours)):
    index_level = int(hierachy[0][i][1])
    if index_level <= i:
        cnt = contours[i]
        area = cv.contourArea(cnt)
        print(area)
        if area <= threshold_blobs_area:
            cv.drawContours(img1, [cnt], -1, 0, -1, 1)
cv.imshow("result", img1)
canny = cv.Canny(img1, 100, 200,)

lines = cv.HoughLines(canny, 1, np.pi / 180, 150)
black = cv.imread("img/black.jpg")
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))

    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv.line(black, (x1, y1), (x2, y2), (0, 0, 255), 2)

dilation = cv.dilate(black, kernal, iterations = 3)
erosion = cv.erode(black, kernal, iterations=3)
opening = cv.morphologyEx(black, cv.MORPH_OPEN, kernal)
closing = cv.morphologyEx(black, cv.MORPH_CLOSE, kernal)
mg = cv.morphologyEx(black, cv.MORPH_GRADIENT, kernal)
th = cv.morphologyEx(black, cv.MORPH_TOPHAT, kernal)

titles = ['image', 'mask', 'dilation', 'erosion', 'opening', 'closing', 'mg', 'th']
images = [img, mask, dilation, erosion, opening, closing, mg , th]



for i in range(8):
    plt.subplot(2, 4, i+1), plt.imshow(images[i], 'Accent')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
cv.imshow("img", black)
cv.imshow("th2", canny)
plt.show()
cv.waitKey(0)
cv.destroyAllWindows()
# cv.imshow("feed", mask)
# cv.waitKey(0)
# cv.destroyAllWindows()


