import numpy as np
import cv2 as cv
import imutils

cap = cv.VideoCapture('AM.mov', cv.IMREAD_GRAYSCALE)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    greenLower = np.array([21, 90, 130])
    greenUpper = np.array([35, 255, 255])

    gaussBlur = cv.GaussianBlur(frame1, (11, 11), 0)
    hsv = cv.cvtColor(gaussBlur, cv.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv.inRange(hsv, greenLower, greenUpper)
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
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
            cv.circle(frame1, (int(x), int(y)), int(radius),
                      (0, 255, 255), 2)
            cv.circle(frame1, center, 5, (0, 0, 255), -1)

    cv.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(40) == 27:
        break

#cv.imshow("feed", frame1)
cv.destroyAllWindows()
cap.release()

