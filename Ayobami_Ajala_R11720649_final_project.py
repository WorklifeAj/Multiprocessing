import argparse
import re
from pathlib import Path
import os
from typing import List
from copy import deepcopy
import multiprocessing

class RowOfStringMatrix:
    def __init__(self, string_row: List[str]):
        self.string_row = string_row
    
    def get_char(self, index: int) -> str:
        """Safely get a character from the row."""
        if 0 <= index < len(self.string_row):
            return self.string_row[index]
        return ''  

class CustomStringMatrix:
    def __init__(self, string_matrix: List[RowOfStringMatrix]):
        self.string_matrix = string_matrix
        
    def get_cell_neighbors_sum(self, row_index: int, col_index: int) -> int:
        """Calculate the sum of the neighbors' values for a given cell."""
        result = 0
        a = ord('a')

        for row in range(max(0, row_index - 1), min(len(self.string_matrix), row_index + 2)):
            for col in range(max(0, col_index - 1), min(len(self.string_matrix[row].string_row), col_index + 2)):
                result += ord(self.string_matrix[row].get_char(col)) - a

        result -= ord(self.string_matrix[row_index].get_char(col_index)) - a
        return result

    def update_matrix_parallel(self, pool):
        """Update the matrix in parallel using multiprocessing."""
        tasks = [(self, row_index) for row_index in range(len(self.string_matrix))]

        # Process each row in parallel and collect the updated rows
        updated_rows = pool.map(process_row, tasks)

        # Reassemble the matrix from the updated rows
        for row_index, updated_row in enumerate(updated_rows):
            self.string_matrix[row_index].string_row = updated_row

def process_row(args):
    """Process a single row using its index."""
    matrix, row_index = args
    updated_row = []

    for col_index, cell_value in enumerate(matrix.string_matrix[row_index].string_row):
        neighbors_sum = matrix.get_cell_neighbors_sum(row_index, col_index)
        if not is_prime(neighbors_sum):
            a = ord('a')
            offset = neighbors_sum % 2 + 1
            new_value = (ord(cell_value) - a + offset) % 3
            updated_row.append(chr(new_value + a))
        else:
            updated_row.append(cell_value)

    return updated_row

def validate_arguments():
    """Validates command line arguments."""
    description = "Validate and process input arguments."
    parser = argparse.ArgumentParser(description=description)

    # Adding arguments
    parser.add_argument('-i', '--input', type=Path, help='Input file path', required=True)
    parser.add_argument('-s', '--seed', type=str, help='Seed string', required=True)
    parser.add_argument('-o', '--output', type=Path, help='Output file path', required=True)
    parser.add_argument('-p', '--processes', type=int, default=1, help='Number of processes', required=False)

    args = parser.parse_args()

    # Validations
    if not args.input.is_file():
        raise argparse.ArgumentTypeError(f"The input file {args.input} was not found.")

    if not re.fullmatch(r"[abc]+", args.seed):
        raise argparse.ArgumentTypeError(f"The seed string '{args.seed}' should only contain 'a', 'b', or 'c'.")

    output_dir = args.output.parent
    if not output_dir.exists() or not output_dir.is_dir():
        raise argparse.ArgumentTypeError(f"The directory for the output file {args.output} does not exist.")

    if args.processes < 1:
        raise argparse.ArgumentTypeError(f"The number of processes must be a positive integer, got {args.processes}.")
      
    return args  

def read_input_file(file_path: Path) -> str:
    """Read the input file and return its content."""
    with file_path.open(mode='r') as file:
        input_string = file.read().strip()
    return input_string

def square_matrix(dimension: int) -> CustomStringMatrix:
    """Create a square matrix of the given dimension."""
    rows = [RowOfStringMatrix([''] * dimension) for _ in range(dimension)]
    return CustomStringMatrix(rows)

def fill_custom_matrix_with_seed(seed: str, matrix: CustomStringMatrix) -> CustomStringMatrix:
    """Fill the custom matrix with the seed string."""
    seed_length = len(seed)
    for i, row in enumerate(matrix.string_matrix):
        for j in range(len(row.string_row)):
            char_index = (i * len(row.string_row) + j) % seed_length
            row.string_row[j] = seed[char_index]
    return matrix

def is_prime(x: int) -> bool:
    """Check if a number is prime."""
    return x in {2, 3, 5, 7, 11, 13, 17}

def column_sum(encrypted: CustomStringMatrix, col_number: int) -> int:
    """Calculate the sum of a column in the encrypted matrix."""
    current_sum = 0
    for row in encrypted.string_matrix:
        current_sum += ord(row.get_char(col_number)) - ord('a')
    return current_sum

def decryptLetter(letter: str, rotationValue: int) -> str:
    """Decrypt a letter using the given rotation value."""
    rotationString = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "
    currentPosition = rotationString.find(letter)
    return rotationString[(currentPosition + rotationValue) % 95]

def encryptLetter(letter: str, rotationValue: int) -> str:
    """Encrypt a letter using the given rotation value."""
    rotationString = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ "
    currentPosition = rotationString.find(letter)
    return rotationString[(currentPosition - rotationValue) % 95]


def main():
    print("Project :: R11702649")

    # Parsing the arguments
    args = validate_arguments()

    # Reading the input string
    input_string = read_input_file(args.input)
    input_string_length = len(input_string)

    # Creating and seeding the matrix
    matrix = square_matrix(input_string_length)
    seeded_matrix = fill_custom_matrix_with_seed(args.seed, matrix)

    # Set up multiprocessing pool
    with multiprocessing.Pool(processes=args.processes) as pool:
        # Updating the matrix 100 times in parallel
        for step in range(100):
            seeded_matrix.update_matrix_parallel(pool)

    # Decrypting each letter from the input string
    decrypted_string = ""
    for i in range(input_string_length):
        current_column_sum = column_sum(seeded_matrix, i)

        # Decrypting the letter
        decrypted_letter = decryptLetter(input_string[i], current_column_sum)
        decrypted_string += decrypted_letter

    # Writing the decrypted string to the output file
    with open(args.output, 'w') as output_file:
        output_file.write(decrypted_string)

if __name__ == "__main__":
    main()

