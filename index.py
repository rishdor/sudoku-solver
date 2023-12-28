import cv2
import numpy as np
import tensorflow as tf
import sudoku_solving as ss

model = tf.keras.models.load_model('model.h5')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)
    return cv2.resize(thresh, (300,300), interpolation = cv2.INTER_AREA)

def find_contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]

def transform_perspective(img, contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    corners = sorted(approx.ravel().reshape(-1, 2), key=lambda x: x[1])
    ordered_corners = np.float32(sorted(corners[:2], key=lambda x: x[0]) + sorted(corners[2:], key=lambda x: x[0], reverse=True))
    input_coords = np.float32(ordered_corners)
    output_coords = np.float32([[0,0], [299,0], [299,299], [0,299]])
    matrix = cv2.getPerspectiveTransform(input_coords, output_coords)
    return cv2.resize(cv2.warpPerspective(img, matrix, (300,300)), (270, 270), interpolation = cv2.INTER_AREA)

def split_image(img):
    rows = np.vsplit(img, 9)
    cells = [cell for r in rows for cell in np.hsplit(r, 9)]
    return cells

def predict_number(cell):
    cell = cv2.resize(cell, (28, 28))
    cell = cv2.cvtColor(cell, cv2.COLOR_GRAY2RGB)
    cell = cell.reshape(1, 28, 28, 3)
    return np.argmax(model.predict(cell)[0])

def read_sudoku(cells):
    return np.array([predict_number(cell) for cell in cells]).reshape(9, 9)

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

def solve_and_display_sudoku(sudoku, img):
    solution = ss.solve_sudoku(sudoku)
    if solution:
        solved_img = overlay_solution(img.copy(), sudoku)
        cv2.imshow('Sudoku solution', solved_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('No solution found')

def main():
    preprocessed_img = preprocess_image('img\sudoku5.png')
    sudoku_contour = find_contours(preprocessed_img)
    sudoku_grid = transform_perspective(preprocessed_img, sudoku_contour)
    cells = split_image(sudoku_grid)
    sudoku = read_sudoku(cells)
    print(sudoku)
    solve_and_display_sudoku(sudoku, sudoku_grid)

if __name__ == "__main__":
    main()
