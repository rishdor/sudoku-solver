# Sudoku Solver

This is a Sudoku solver that uses a backtracking algorithm to solve the puzzle and CNN to recognize digits from images of Sudoku puzzles.

## Features

- Solves Sudoku puzzles using a backtracking algorithm.
- Recognizes digits from images of Sudoku puzzles using a trained CNN.
- Overlays the solution onto the original image of the Sudoku puzzle.

## How It Works

*Original Sudoku Image:*
  
![resized_original](https://github.com/rishdor/sudoku-solver/assets/66086647/0895aea5-a9ce-49ad-821c-2d0a0af633c6)

- The script first preprocesses the input image to prepare it for digit recognition. It then uses contour detection and perspective transformation to isolate the Sudoku grid and split it into individual cells.

![processed](https://github.com/rishdor/sudoku-solver/assets/66086647/e34ab3a7-7a45-4996-9dd3-2fe8868315f0)

- The script uses a trained CNN to recognize the digits in each cell. The recognized digits are used to construct the Sudoku puzzle.

![np_array](https://github.com/rishdor/sudoku-solver/assets/66086647/2b519037-2c36-4238-afa6-ddd56c6a16db)

- The script solves the Sudoku puzzle using a backtracking algorithm. It then overlays the solution onto the original image of the Sudoku puzzle.

![solved](https://github.com/rishdor/sudoku-solver/assets/66086647/5e101f76-5053-4004-b704-d10bc5102cbe)

## Model

The model is a Convolutional Neural Network (CNN) that has been trained to recognize digits from images. The model was trained using the TensorFlow and Keras libraries. Here is the evaluation of the model:

```plaintext
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        60
           1       0.99      1.00      0.99        69
           2       1.00      1.00      1.00        69
           3       1.00      0.98      0.99        66
           4       1.00      1.00      1.00        71
           5       1.00      0.98      0.99        64
           6       1.00      1.00      1.00        57
           7       1.00      1.00      1.00        63
           8       0.98      1.00      0.99        57
           9       1.00      1.00      1.00        53

    accuracy                           1.00       629
   macro avg       1.00      1.00      1.00       629
weighted avg       1.00      1.00      1.00       629
```

## Dataset

The model was trained on the [Printed Digits Dataset](https://www.kaggle.com/datasets/kshitijdhama/printed-digits-dataset).

## Limitations

This script assumes that the input image is a clear, straight-on view of a Sudoku puzzle where the puzzle grid is the largest contour in the image. If the perspective of the image is significantly skewed or distorted, the program may encounter difficulties in accurately recognizing the digits.

---

*This program has been tested and works with all the images in the `img` folder in this repository.*
