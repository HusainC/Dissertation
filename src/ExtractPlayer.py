import cv2
import numpy as np

# Sample video to perform player detection.
cap = cv2.VideoCapture('../resources/videos/shade.mp4')
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Read the video frame by frame and check for moving objects using contours.
while cap.isOpened():
    # Find difference between two frames.
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use thresholding to differentiate between objects of different intensity.
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # OpenCV method used to find contours from the morphological operation performed.
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop too go through the contours frame by frame and and find players.
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 500:
            continue

        # Draw a rectangle around the moving player.
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

    # Show frame1 and replace the frame for the next frame.
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()