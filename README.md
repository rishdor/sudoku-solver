# Sudoku Solver

This is a Sudoku solver that uses a backtracking algorithm to solve the puzzle and a Convolutional Neural Network (CNN) to recognize digits from images of Sudoku puzzles.

## Features

- Solves Sudoku puzzles using a backtracking algorithm.
- Recognizes digits from images of Sudoku puzzles using a trained CNN.
- Overlays the solution onto the original image of the Sudoku puzzle.

## How It Works

The script first preprocesses the input image to prepare it for digit recognition. It then uses contour detection and perspective transformation to isolate the Sudoku grid and split it into individual cells.

The script uses a trained CNN to recognize the digits in each cell. The recognized digits are used to construct the Sudoku puzzle.

The script solves the Sudoku puzzle using a backtracking algorithm. It then overlays the solution onto the original image of the Sudoku puzzle.

## Limitations

This script assumes that the input image is a clear, straight-on view of a Sudoku puzzle where the puzzle grid is the largest contour in the image. If the perspective of the image is significantly skewed or distorted, the program may encounter difficulties in accurately recognizing the digits.

## Acknowledgements

This project uses the TensorFlow and Keras libraries for digit recognition, and the OpenCV library for image processing.
