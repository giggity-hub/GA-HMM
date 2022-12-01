from drs import drs
import numpy

# Returns an array with the cutpoints corresponding to the rows in the hmm
def get_cut_points(num_states, num_symbols):
    # bei 0 zu cutten wäre bissl dumm weil dadurch hat man ja die gleiche fitness
    # Der erste cut point ist hinter den startwahrscheinlichkeiten
    # cut_points = [] 
    offset = num_states
    cut_points = [offset + i*num_states for i in range(num_states+1)]
    offset = cut_points[-1]
    cut_points+= [offset + num_symbols*i for i in range(1,num_states)]
    return cut_points



def normalize_vector(prob_array):
    row_sums = numpy.sum(prob_array, axis=0, keepdims=True)
    return prob_array / row_sums


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
