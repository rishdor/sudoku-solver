# put a number in an empty cell
# check if that number can be in a cell (row, column, square)
# if it can, try solving the rest of the puzzle with this number
# if failed, try solving for the next number

def solve_sudoku(sudoku):
    if not find_empty_cell(sudoku):
        return True

    row, col = find_empty_cell(sudoku)
    for num in range(1, 10):
        if is_valid(sudoku, row, col, num):
            sudoku[row][col] = num

            if solve_sudoku(sudoku):
                return True

            sudoku[row][col] = 0

    return False

def find_empty_cell(sudoku):
    for row in range(9):
        for col in range(9):
            if sudoku[row][col] == 0:
                return row, col
    return None

# check if the number can be in the cell
def is_valid(sudoku, row, col, num):
    return (
        is_valid_row(sudoku, row, num) and
        is_valid_column(sudoku, col, num) and
        is_valid_square(sudoku, row - row % 3, col - col % 3, num)
    )
    
# check if the number is not in the row
def is_valid_row(sudoku, row, num):
    for col in range(9):
        if sudoku[row][col] == num:
            return False
    return True

# check if the number is not in the column
def is_valid_column(sudoku, col, num):
    for row in range(9):
        if sudoku[row][col] == num:
            return False
    return True

# check if the number is not in the square
def is_valid_square(sudoku, start_row, start_col, num):
    for row in range(3):
        for col in range(3):
            if sudoku[row + start_row][col + start_col] == num:
                return False
    return True

sudoku = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

solve_sudoku(sudoku)
