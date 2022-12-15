from drs import drs
import numpy

def normalize_vector(prob_array):
    row_sums = numpy.sum(prob_array, axis=0, keepdims=True)
    return prob_array / row_sums

def normalize_matrix(prob_matrix):
    return [normalize_vector(row) for row in prob_matrix]

# Returns an Array which is row stochastic
def rand_stochastic_vector(length):
    upper_bounds = [1]*length
    lower_bounds = [0]*length
    sumu = 1
    return drs(length, sumu, upper_bounds, lower_bounds)

# shape= (rows, columns)
def rand_stochastic_matrix(row_count, col_count):
    # row_count, col_count = shape 
    return [rand_stochastic_vector(col_count) for i in range(row_count)]
