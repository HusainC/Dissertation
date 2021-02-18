import numpy as np
import cv2 as cv
import imutils

cap = cv.VideoCapture('videos/tm.mp4', cv.IMREAD_GRAYSCALE)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():

    params = cv.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.maxArea = 1000

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
    keypoints = detector.detect(frame1)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob
    for i in keypoints:
        print(i.pt)

    print(keypoints)

    img = cv.drawKeypoints(frame1, keypoints, np.array([]), (0, 0, 255),
                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv.imshow("feed", img)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv.waitKey(40) == 27:
        break

#cv.imshow("feed", frame1)
cv.destroyAllWindows()
cap.release()

