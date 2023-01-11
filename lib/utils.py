from drs import drs
import numpy

def normalize_vector(prob_array: numpy.array) -> numpy.array:
    row_sums = numpy.sum(prob_array, axis=0, keepdims=True)
    return prob_array / row_sums

def normalize_matrix(prob_matrix: numpy.ndarray) -> numpy.ndarray:
    # dimensions have to be kept. Otherwise will result in broadcasting error
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    return prob_matrix / row_sums


# Returns an Array which is row stochastic
def rand_stochastic_vector(length: int) -> numpy.array:
    upper_bounds = [1]*length
    lower_bounds = [0]*length
    sumu = 1
    res =  drs(length, sumu, upper_bounds, lower_bounds)
    return numpy.array(res)

# shape= (rows, columns)
def rand_stochastic_matrix(row_count: int, col_count: int) -> numpy.ndarray :
    # row_count, col_count = shape 
    arrs =  [rand_stochastic_vector(col_count) for i in range(row_count)]
    return numpy.stack(arrs)
