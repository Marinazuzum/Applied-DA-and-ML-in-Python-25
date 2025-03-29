import numpy as np


import numpy as np

# Define the function to create an array
def create_array(array_length):
    """
    Input:
        - array_length:     int; specifies length of array
    Return:
        - rand_array:       numpy array; dim: (array_length, )
    Function:
        - Creates an array with random integers in the range [0, 20]
    """
    rand_array = np.random.randint(0, 21, size=array_length)  # Random integers from 0 to 20
    return rand_array

# Test the function by creating an array of length 10
array_length = 10
result = create_array(array_length)
print(result)



import numpy as np

# Function to append two arrays
def append_array(array1, array2):
    """
    Input:
        - array1:   numpy array; first array to append
        - array2:   numpy array; second array to append, same dimensions as array1
    Return:
        - array3:   numpy array; dim(2 * length(array1)); the result of appending array1 to array2
    Function:
        - Appends array1 to array2 (mind the order!).
    """
    # Check if the arrays have the same shape
    if array1.shape != array2.shape:
        raise ValueError("Input arrays must have the same shape.")
    
    # Append the arrays
    array3 = np.append(array1, array2, axis=0)
    return array3

# Test the function with two arrays of the same shape
array1 = np.random.randint(0, 21, size=(5, 3))  # Array with shape 5x3
array2 = np.random.randint(0, 21, size=(5, 3))  # Array with shape 5x3

# Call the function
result = append_array(array1, array2)

# Print the result
print(result)



import numpy as np

# Function to sort array in ascending and descending order, then sum the arrays
def sort_array(array):
    """
    Input:
        - array:        numpy array; dim (20, ); the array to be sorted
    Return:
        - array_sym:    numpy array; dim (20, ); the sum of the sorted arrays
    Function:
        - Sorts the array in ascending order and stores the result in array_asc
        - Sorts the array in descending order
        - Adds both arrays and returns the result as array_sym
    """
    # Check if the array has the shape (20,)
    if array.shape != (20,):
        raise ValueError("Input array must have shape (20,).")
    
    # Step 1: Sort the array in ascending order
    array_asc = np.sort(array)
    
    # Step 2: Sort the array in descending order
    array_desc = np.sort(array)[::-1]  # We can reverse the sorted array to get descending order
    
    # Step 3: Add the ascending and descending arrays
    array_sym = array_asc + array_desc
    
    return array_sym

# Test the function with a random array of shape (20,)
array = np.random.randint(0, 21, size=(20,))  # Generate a random array with 20 elements

# Call the function
result = sort_array(array)

# Print the result
print("Input Array:", array)
print("Resulting Array:", result)



import numpy as np

# Function to scale values so that the highest value becomes 20, and round to 2 decimals
def scale(array_in):
    """
    Input:
        - array_in:     numpy array; dim (20, ); the array to be scaled
    Return:
        - array_scaled: numpy array; dim (20, ); the scaled and rounded array
    Function:
        - Scales all values by the same factor to make the highest value exactly 20
        - Rounds all values to 2 decimal places
    """
    # Check if the array has the shape (20,)
    if array_in.shape != (20,):
        raise ValueError("Input array must have shape (20,).")
    
    # Step 1: Find the maximum value in the array
    max_value = np.max(array_in)
    
    # Step 2: Calculate the scaling factor
    scaling_factor = 20 / max_value
    
    # Step 3: Scale all values by the scaling factor
    array_scaled = array_in * scaling_factor
    
    # Step 4: Round all values to 2 decimal places
    array_scaled = np.round(array_scaled, 2)
    
    return array_scaled

# Test the function with a random array of shape (20,)
array_in = np.random.randint(0, 41, size=(20,))  # Generate a random array



import numpy as np

# Function to sort columns indirectly by the first row
def indirect_sort(array_in):
    """
    Input:
        - array_in:    numpy array; dim (20, ); the array to be reshaped and sorted
    Return:
        - matrix:      numpy array; dim (2, 10); the sorted matrix with the second row as idx
    Function:
        - Reshapes the input array to (2, 10)
        - Creates an idx array with integers from 1 to 10 in arbitrary order
        - Inserts the idx array as the second row
        - Sorts the columns of the matrix based on the first row and returns the sorted matrix
    """
    # Check if the array has the shape (20,)
    if array_in.shape != (20,):
        raise ValueError("Input array must have shape (20,).")
    
    # Step 1: Reshape the input array to (2, 10)
    matrix = array_in.reshape(2, 10)
    
    # Step 2: Create the idx array with integers from 1 to 10 in arbitrary order
    idx = np.random.permutation(10) + 1  # Random permutation of 



import numpy as np

# Function to compute the sum of all elements, rows, and columns of a matrix
def matrix_sum(matrix_in):
    """
    Input:
        - matrix_in:    2D numpy array with arbitrary dimensions
    Returns:
        - sum_all:      sum of all elements of the matrix
        - sum_row:      sum of each row
        - sum_column:   sum of each column
    Function:
        - computes the sum of all elements of the matrix, sums for rows and columns
    """
    # Sum all elements in the matrix
    sum_all = np.sum(matrix_in)
    
    # Sum by rows (axis=1 sums along rows)
    sum_row = np.sum(matrix_in, axis=1)
    
    # Sum by columns (axis=0 sums along columns)
    sum_column = np.sum(matrix_in, axis=0)
    
    return sum_all, sum_row, sum_column

# Example usage of the function
if __name__ == "__main__":
    # Generate a random 2D matrix (e.g., 5x4)
    matrix_in = np.random.randint(0, 11, size=(5, 4))  # A 5x4 matrix with values between 0 and 10
    
    # Call the function to compute the sums
    sum_all, sum_row, sum_column = matrix_sum(matrix_in)

    # Print the results
    print("Input Matrix:\n", matrix_in)
    print("Sum of all elements:", sum_all)
    print("Sum of each row:", sum_row)
    print("Sum of each column:", sum_column)
