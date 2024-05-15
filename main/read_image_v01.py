import csv
import random

def normalize_pixel_value(value):
    """Normalize the pixel value to be between 0 and 1."""
    return float(value) / 255.0

def read_random_row_from_csv(file_path):
    """Read a random row between 1 and 10 from the CSV file and return as a list of normalized floats, excluding the first value."""
    with open(file_path, newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        random_row_index = random.randint(1, 10) - 1  # Randomly select a row index between 0 and 9 (corresponding to rows 1 to 10)
        random_row = reader[random_row_index]  # Get the selected row
        normalized_data = [normalize_pixel_value(value) for value in random_row[1:]]  # Skip the first value and normalize the rest
        return normalized_data

def write_data_to_file(file_path, data):
    """Write the data to a file with ',\n\r' as the separator."""
    with open(file_path, 'w') as file:
        for value in data:
            file.write(f"{value},\n\r")

def main():
    # Read a random row between 1 and 10 from src.csv
    normalized_data = read_random_row_from_csv('data/mnist_test.csv')

    # Write the normalized data to b.dat
    write_data_to_file('out/a05.dat', normalized_data)

if __name__ == "__main__":
    main()
