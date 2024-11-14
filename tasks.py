import numpy as np

# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array,
# kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no
# padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
# -----------------------------------------------

def compute_output_size_1d(input_array: np.ndarray,
                           kernel_array: np.ndarray) -> int:
    """
    Compute length of one-dimensional array given an input and kernel array.

    Args:
        input_array (np.ndarray): 1D Input array.
        kernel_array (np.ndarray): 1D Kernel array.

    Returns:
        int: Output length of 1D array.
    """

    return input_array.size - kernel_array.size + 1


# -----------------------------------------------
# Example:
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(compute_output_size_1d(input_array, kernel_array))


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a
# one-dimensional kernel array (kernel_array) and returns their convolution
# (no padding, stride 1).

# Your code here:
# -----------------------------------------------

def convolve_1d(input_array: np.ndarray,
                kernel_array: np.ndarray) -> np.ndarray:
    """
    Convolve input array with kernel array with stride 1 and no padding.

    Args:
        input_array (np.ndarray): 1D input array.
        kernel_array (np.ndarray): 1D kernel array.

    Returns:
        np.ndarray: Convolved array.
    """

    output_array = np.zeros(compute_output_size_1d(input_array,
                                                   kernel_array))
    kernel_len = kernel_array.size
    for i in range(output_array.size):
        output_array[i] = np.dot(input_array[i: i + kernel_len],
                                 (kernel_array))

    return output_array


# -----------------------------------------------
# Another tip: write test cases like this,
# so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix,
# kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of
# both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be
# computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
# -----------------------------------------------


def compute_output_size_2d(input_matrix: np.ndarray,
                           kernel_matrix: np.ndarray) -> tuple:
    """
    Compute output dimensions after convolution of input matrix with kernel
    matrix.

    Args:
        input_matrix (np.ndarray): 2D Input matrix.
        kernel_matrix (np.ndarray): 2D Kernel matrix.

    Returns:
        tuple: Dimensions after convolution.
    """

    return (input_matrix.shape[0] - kernel_matrix.shape[0] + 1,
            input_matrix.shape[1] - kernel_matrix.shape[1] + 1)
# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two
# matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions
# you made above.

# Your code here:
# -----------------------------------------------
def convolute_2d(input_matrix: np.ndarray,
                 kernel_matrix: np.ndarray) -> np.ndarray:
    """
    Compute the convolution of and input matrix and a kernel matrix with stride
    1 and no padding.

    Args:
        input_matrix (np.ndarray): Input matrix.
        kernel_matrix (np.ndarray): Kernel matrix.

    Returns:
        np.ndarray: Convolved matrix.
    """

    output_matrix = np.zeros(compute_output_size_2d(input_matrix,
                                                    kernel_matrix))
    kernel_len_x = kernel_matrix.shape[0]
    kernel_len_y = kernel_matrix.shape[1]

    for i in range(output_matrix.shape[0]):
        for j in range(output_matrix.shape[1]):
            output_matrix[i, j] = np.dot(input_matrix[i: i + kernel_len_x,
                                                      j: j + kernel_len_y],
                                         kernel_matrix)
    return output_matrix

# -----------------------------------------------
