import cv2
import numpy as np
import tensorflow as tf
import sudoku_solving as ss
model = tf.keras.models.load_model('model.h5')

# preprocess the image

def preprocess_image(img):
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (5, 5), 0)
    thresh=cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    resized = cv2.resize(thresh, (300,300), interpolation = cv2.INTER_AREA)
    
    return resized

img = cv2.imread('img\sudoku1.png')
preprocessed_img = preprocess_image(img)

# contour detection

contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
sudoku_contour = contours[0]

# perspective transformation

epsilon = 0.01 * cv2.arcLength(sudoku_contour, True)
approx = cv2.approxPolyDP(sudoku_contour, epsilon, True)

corners = approx.ravel().reshape(-1, 2)
corners = sorted(corners, key=lambda x: x[1])

left = corners[:2]
right = corners[2:]

left = sorted(left, key=lambda x: x[0])
right = sorted(right, key=lambda x: x[0], reverse=True)

ordered_corners = np.float32(left + right)

input_coords = np.float32(ordered_corners)
output_coords = np.float32([[0,0], [299,0], [299,299], [0,299]])

matrix = cv2.getPerspectiveTransform(input_coords, output_coords)
transformed_img = cv2.warpPerspective(preprocessed_img, matrix, (300,300))
sudoku_grid = cv2.resize(transformed_img, (270, 270), interpolation = cv2.INTER_AREA)

# cell segmentation

def split_image(img):
    rows = np.vsplit(img, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells

cells = split_image(sudoku_grid)

# number recognition

def predict_number(cell):
    cell = cv2.resize(cell, (28, 28))
    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
    cell = cell.reshape(1, 28, 28, 3)
    
    predictions = model.predict(cell)
    predicted_digit = np.argmax(predictions[0])

    return predicted_digit

def read_sudoku(cells):
    sudoku = []
    for cell in cells:
        digit = predict_number(cell)
        sudoku.append(digit)
    sudoku = np.array(sudoku).reshape(9, 9)
    return sudoku

sudoku = read_sudoku(cells)
print(sudoku)

# overlay the solution

def overlay_solution(img, sudoku):
    cell_size = img.shape[0] // 9
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    for i in range(9):
        for j in range(9):
            if sudoku[i][j] != 0:
                x = j * cell_size
                y = i * cell_size
                number = str(sudoku[i][j])
                text_size, _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                text_x = x + (cell_size - text_size[0]) // 2
                text_y = y + (cell_size + text_size[1]) // 2
                cv2.putText(img, number, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return img

# sudoku solving

solution = ss.solve_sudoku(sudoku)
if solution:
    solved_img = sudoku_grid.copy()
    solved_img = overlay_solution(solved_img, sudoku)
    cv2.imshow('Sudoku solution', solved_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('no solution found')
