import cv2 as cv
import numpy as np
import imutils

img = cv.imread('img/rog.jpg')
#img2 = cv.pyrDown(img)

#gblur = cv.GaussianBlur(img2, (5,5), 0)

#hsv = cv.cvtColor(gblur, cv.COLOR_BGR2HSV)
greenLower = np.array([0, 190, 114])
greenUpper = np.array([53, 255, 255])

#mask = cv.inRange(hsv, l_b, u_b)

#res = cv.bitwise_and(img2, img2, mask = mask)

#cv.imshow("res", res)


blurred = cv.GaussianBlur(img, (11, 11), 0)
hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask = cv.inRange(hsv, greenLower, greenUpper)
mask = cv.erode(mask, None, iterations=2)
mask = cv.dilate(mask, None, iterations=2)

cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
print(cnts)
cnts = imutils.grab_contours(cnts)
center = None
# only proceed if at least one contour was found
if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv.contourArea)
		((x, y), radius) = cv.minEnclosingCircle(c)
		M = cv.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv.circle(img, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv.circle(img, center, 5, (0, 0, 255), -1)

cv.imshow('Image', img)
cv.imshow('Image GRAY', img)
#cv.imshow('Threshhold', gblur)
cv.waitKey(0)
cv.destroyAllWindows()

