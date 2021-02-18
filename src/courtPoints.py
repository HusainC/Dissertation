import cv2
import numpy as np

def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return Px, Py

def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x2-x1) < delta: # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2-y1) < delta: # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines

def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers

# preprocessing
img = cv2.imread("../resources/img/courtG.jpg")

greenLower = np.array([98, 111, 98])
greenUpper = np.array([120, 181, 152])

gaussBlur = cv2.GaussianBlur(img, (11, 11), 0)
hsv = cv2.cvtColor(gaussBlur, cv2.COLOR_BGR2HSV)
# construct a mask for the color "green", then perform
# a series of dilations and erosions to remove any small
# blobs left in the mask
mask1 = cv2.inRange(hsv, greenLower, greenUpper)
mask2 = cv2.erode(mask1, None, iterations=2)
mask3 = cv2.dilate(mask2, None, iterations=2)
_, mask = cv2.threshold(mask3, 220, 255, cv2.THRESH_BINARY_INV)

kernal = np.ones((5,5), np.uint8)

# dilation = cv.dilate(mask, kernal, iterations = 3)
erosion = cv2.erode(mask, kernal, iterations=3)
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
ret, img1 = cv2.threshold(erosion, 125, 255, cv2.THRESH_BINARY_INV)


contours, hierachy = cv2.findContours(img1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

threshold_blobs_area = 5500

for i in range(1, len(contours)):
    index_level = int(hierachy[0][i][1])
    if index_level <= i:
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print(area)
        if area <= threshold_blobs_area:
            cv2.drawContours(img1, [cnt], -1, 0, -1, 1)
cv2.imshow("result", img1)
canny = cv2.Canny(img1, 100, 200,)

# run the Hough transform
lines = cv2.HoughLines(canny, rho=1, theta=np.pi/180, threshold=150)

# segment the lines
delta = 10
h_lines, v_lines = segment_lines(lines, delta)

# draw the segmented lines
#lines = cv2.HoughLines(canny, 1, np.pi / 180, 150)
black = cv2.imread("../resources/img/black.jpg")
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
    cv2.line(black, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv2.imshow("Segmented Hough Lines", black)
cv2.waitKey(0)
cv2.imwrite('../resources/img/hough.png', black)

# find the line intersection points
Px = []
Py = []
for h_line in h_lines:
    for v_line in v_lines:
        px, py = find_intersection(h_line, v_line)
        Px.append(px)
        Py.append(py)

# draw the intersection points
intersectsimg = img.copy()
for cx, cy in zip(Px, Py):
    cx = np.round(cx).astype(int)
    cy = np.round(cy).astype(int)
    color = np.random.randint(0,255,3).tolist() # random colors
    cv2.circle(intersectsimg, (cx, cy), radius=2, color=color, thickness=-1) # -1: filled circle

cv2.imshow("Intersections", intersectsimg)
cv2.waitKey(0)
cv2.imwrite('../resources/img/intersections.png', intersectsimg)

# use clustering to find the centers of the data clusters
P = np.float32(np.column_stack((Px, Py)))
nclusters = 4
centers = cluster_points(P, nclusters)
print(centers)

# draw the center of the clusters
for cx, cy in centers:
    cx = np.round(cx).astype(int)
    cy = np.round(cy).astype(int)
    cv2.circle(img, (cx, cy), radius=4, color=[0,0,255], thickness=-1) # -1: filled circle

cv2.imshow("Center of intersection clusters", img)
cv2.waitKey(0)
cv2.imwrite('corners.png', img)