from __future__ import print_function

import PIL
from PIL import Image
import cv2 as cv
import numpy as np
import argparse


# Parse the video to perform background subtraction.
from src import ColourUtil

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()

if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

# Video used to perform ball detection.
capture = cv.VideoCapture('../resources/videos/shade.mp4')
kernal = np.ones((1, 1), np.uint8)

# check if video is opened and running without errors
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

# loop to go through each frame of the video.
while True:
    ret, frame = capture.read()

    # Check performed to make sure frame is not null.
    if frame is None:
        break

    # Get an RGB format of the frame.
    red_image = PIL.Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Setting the params for detection of the blob(tennis ball).
    params = cv.SimpleBlobDetector_Params()

    # Filter by Area
    params.filterByArea = True
    params.minArea = 20

    # Filter by Colour
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
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv.SimpleBlobDetector(params)
    else:
        detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs.
    # perform dilation on fgMask to get rid of small noises on screen.
    dilation = cv.dilate(fgMask, kernal, iterations=1)
    keypoints = detector.detect(dilation)
    # List to store all the objects recognized as a ball.
    finalKeypoint = []

    # List to store the balls location in the previous frame
    prevFrame = []

    # Loop to run through the keypoints detected and filter through objects similar to a tennis ball.
    for i in keypoints:
        x1 = i.pt[0]
        x2 = i.pt[1]
        currentFrame = [x1, x2]

        if prevFrame:
            print(prevFrame)
            print("fhello ", checkPrevFrame(prevFrame, currentFrame))
            prevFrame = currentFrame
        else:
            prevFrame.append(x1)
            prevFrame.append(x2)

        rgb_pixel_value = ColourUtil.ColourCheck.get_pixels(red_image, x1, x2)
        my_color = (rgb_pixel_value[0], rgb_pixel_value[1], rgb_pixel_value[2])

        differences = [[ColourUtil.ColourCheck.color_difference(my_color, target_value), target_name]
                       for target_name, target_value in
                       ColourUtil.ColourCheck.TARGET_COLORS.items()]
        differences.sort()  # sorted by the first element of inner lists
        my_color_name = differences[0][1]

        # Check if colour is Yellow or Green.
        if my_color_name == 'Yellow' or my_color_name == 'Green':
            print(my_color_name)
            finalKeypoint.append(i)
            print(rgb_pixel_value)
            print(i)

    # Draw detected blobs as red circles.
    keypointsDetected = cv.drawKeypoints(fgMask, finalKeypoint, np.array([]), (0, 0, 255),
                             cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow('Frame', frame)
    cv.imshow("dialte", dilation)
    cv.imshow('keypoints', keypointsDetected)
    cv.imshow('FG Mask', fgMask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

    def checkPrevFrame(preFrame, currentFrame):
        distancex = currentFrame[0] - prevFrame[0]
        distancey = currentFrame[1] - prevFrame[1]
        distance = [distancex, distancey]
        return distance

