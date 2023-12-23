import cv2 as cv
import numpy as np

# preprocess the image

img = cv.imread('img\sudoku5.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)

thresh = cv.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

resized = cv.resize(thresh, (300, 300), interpolation = cv.INTER_AREA)

# contour detection

contours, _ = cv.findContours(resized, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv.contourArea, reverse=True)
sudoku_contour = contours[0]

resized = cv.cvtColor(resized, cv.COLOR_GRAY2BGR)
cv.polylines(resized, [sudoku_contour], True, (0,255,0), 2)
cv.imshow('Resized with contour', resized)
cv.waitKey(0)
cv.destroyAllWindows()

# perspective transformation

# cell segmentation

# number recognition

# sudoku solving

# overlay the solution