from drs import drs
import numpy
from numba import njit

def normalize_vector(prob_array: numpy.array) -> numpy.array:
    row_sums = numpy.sum(prob_array, axis=0, keepdims=True)
    return prob_array / row_sums

def normalize_matrix(prob_matrix: numpy.ndarray) -> numpy.ndarray:
    # dimensions have to be kept. Otherwise will result in broadcasting error
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    return prob_matrix / row_sums


# Returns an Array which is row stochastic
# def rand_stochastic_vector(length: int) -> numpy.array:
#     upper_bounds = [1]*length
#     lower_bounds = [0]*length
#     sumu = 1
#     res =  drs(length, sumu, upper_bounds, lower_bounds)
#     return numpy.array(res)

@njit
def rand_stochastic_vector(length: int) -> numpy.array:
    vector = numpy.random.rand(length)
    normalized_vector = vector / numpy.sum(vector)
    return normalized_vector

# shape= (rows, columns)
@njit()
def rand_stochastic_matrix(row_count: int, col_count: int) -> numpy.ndarray :
    # row_count, col_count = shape 
    matrix = numpy.random.rand(row_count, col_count)
    sum_vector = numpy.sum(matrix, axis=1).reshape((row_count, 1))
    normalized_matrix = matrix / sum_vector
    return normalized_matrix
