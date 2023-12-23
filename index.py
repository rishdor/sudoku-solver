import cv2 as cv
import numpy as np

# preprocess the image

img = cv.imread('img\sudoku5.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)

thresh = cv.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

resized = cv.resize(thresh, (270, 270), interpolation = cv.INTER_AREA)

# contour detection

contours, _ = cv.findContours(resized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv.contourArea, reverse=True)
sudoku_contour = contours[0]

resized = cv.cvtColor(resized, cv.COLOR_GRAY2BGR)
cv.polylines(resized, [sudoku_contour], True, (0,255,0), 2)
# cv.imshow('Resized with contour', resized)
# cv.waitKey(0)
# cv.destroyAllWindows()

# perspective transformation

epsilon = 0.01 * cv.arcLength(sudoku_contour, True)
approx = cv.approxPolyDP(sudoku_contour, epsilon, True)

corners = approx.ravel().reshape(-1, 2)
corners = sorted(corners, key=lambda x: x[1])

left = corners[:2]
right = corners[2:]

left = sorted(left, key=lambda x: x[0])

right = sorted(right, key=lambda x: x[0], reverse=True)

ordered_corners = np.float32(left + right)

input_coords = np.float32(ordered_corners)
output_coords = np.float32([[0,0], [299,0], [299,299], [0,299]])

matrix = cv.getPerspectiveTransform(input_coords, output_coords)

transformed_img = cv.warpPerspective(resized, matrix, (270,270))

# cv.imshow('Transformed Image', transformed_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cell segmentation

def split_image(img):
    rows = np.vsplit(img, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            boxes.append(box)
    return boxes

boxes = split_image(transformed_img)

print(len(boxes))

# number recognition

# sudoku solving

# overlay the solution