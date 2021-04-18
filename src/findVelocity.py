import numpy as np
import cv2 as cv
import imutils
import PIL
from PIL import Image
import cmath

cap = cv.VideoCapture('../resources/videos/ball.mov', cv.IMREAD_GRAYSCALE)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Get the number of frames in the video to calculate velocity.
length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# Record position of the ball in previous frames
prevFrame = []
oldDistance = None
time = 1 / 240
print(time)


# This method uses Pythagoras theorem to find the distance between two points.
def getDistance(prev_frame, current_frame):
    dist = cmath.sqrt(pow((current_frame[0] - prev_frame[0]), 2) + pow((current_frame[1] - prev_frame[1]), 2))
    return dist


# Method to calculate velocity of the object in between two points.
def calculateVelocity(old_distance, new_distance):
    velocity = (new_distance - old_distance) / time
    return velocity


# Going through each frame of the video
while cap.isOpened():
    # Lower and upper bounds of the object have been set to get proof of concept
    greenLower = np.array([21, 90, 130])
    greenUpper = np.array([35, 255, 255])

    gaussBlur = cv.GaussianBlur(frame1, (11, 11), 0)
    hsv = cv.cvtColor(gaussBlur, cv.COLOR_BGR2HSV)

    # Setup a mask to search between an HSV range
    mask = cv.inRange(hsv, greenLower, greenUpper)

    # A series of dilation's and erosion's to remove any small noise
    mask = cv.erode(mask, None, iterations=2)
    mask = cv.dilate(mask, None, iterations=2)

    red_image = PIL.Image.fromarray(cv.cvtColor(frame1, cv.COLOR_BGR2RGB))

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        c = max(cnts, key=cv.contourArea)
        ((x, y), radius) = cv.minEnclosingCircle(c)
        M = cv.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        currentFrame = [int(x), int(y)]
        distance = None
        vel = None

        if prevFrame:
            distance = getDistance(prevFrame, currentFrame)
        else:
            prevFrame = currentFrame.copy()

        if oldDistance is None:
            oldDistance = distance
        else:
            vel = calculateVelocity(oldDistance, distance)

        print(f"distance{distance}")
        print(f"Velocity{vel}")
        prevFrame = currentFrame.copy()
        oldDistance = distance

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

# cv.imshow("feed", frame1)
cv.destroyAllWindows()
cap.release()
