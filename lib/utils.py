from drs import drs
import numpy
from numba import njit
rng = numpy.random.default_rng()
from typing import Tuple

def normalize_vector(prob_array: numpy.array) -> numpy.array:
    row_sums = numpy.sum(prob_array, axis=0, keepdims=True)
    return prob_array / row_sums

def normalize_matrix(matrix: numpy.ndarray) -> numpy.ndarray:
    pass
    # dimensions have to be kept. Otherwise will result in broadcasting error

    # row_sums = prob_matrix.sum(axis=1, keepdims=True)
    # return prob_matrix / row_sums


# Returns an Array which is row stochastic
# def rand_stochastic_vector(length: int) -> numpy.array:
#     upper_bounds = [1]*length
#     lower_bounds = [0]*length
#     sumu = 1
#     res =  drs(length, sumu, upper_bounds, lower_bounds)
#     return numpy.array(res)

@njit
def uniform_rand_stochastic_vector(length: int) -> numpy.array:
    vector = numpy.random.rand(length)
    normalized_vector = vector / numpy.sum(vector)
    return normalized_vector


def conditional_rand_stochastic_matrix(row_count: int, col_count: int) -> numpy.array:
    matrix = numpy.empty((row_count, col_count))

    sum_of_nth_row = numpy.zeros(row_count)

    for i in range(row_count):
        matrix[i, 0] = rng.uniform(0,1)
        sum_of_nth_row[i] += matrix[i, 0]
        for j in range(1, col_count - 1):
            upper_limit = 1-sum_of_nth_row[i]
            matrix[i, j] = rng.uniform(0, upper_limit)
            sum_of_nth_row[i] += matrix[i, j]
        matrix[i,-1] = 1 - sum_of_nth_row[i]
    
    return matrix


# shape= (rows, columns)
@njit()
def uniform_rand_stochastic_matrix(row_count: int, col_count: int) -> numpy.array :
    # row_count, col_count = shape 
    matrix = numpy.random.rand(row_count, col_count)
    sum_vector = numpy.sum(matrix, axis=1).reshape((row_count, 1))
    normalized_matrix = matrix / sum_vector
    return normalized_matrix


