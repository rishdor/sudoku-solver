import cv2 as cv
import numpy as np

# preprocess the image

img = cv.imread('img\sudoku5.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = gray.astype('uint8')

blurred = cv.GaussianBlur(gray, (5, 5), 0)

thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)

resized = cv.resize(thresh, (200, 200), interpolation = cv.INTER_AREA)

normalized = resized / 255.0

cv.imshow('Preprocessed image', resized)
cv.waitKey(0)
cv.destroyAllWindows()

# contour detection

# perspective transformation

# cell segmentation

# number recognition

# sudoku solving

# overlay the solution