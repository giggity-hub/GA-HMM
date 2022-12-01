from ga.crossover import single_row_cutpoint
from hmm.hmm import random_chromosome
import numpy

ERROR_TOLERANCE = 5e-16
def within_tolerance(a,b):
    return abs(a - b) <= ERROR_TOLERANCE

def is_row_stochastic(matrix):
    if matrix.max() > 1 or matrix.min() < 0:
        return False
    if matrix.ndim == 1:
        return within_tolerance(sum(matrix), 1)
    elif matrix.ndim ==2:
        row_sums = matrix.sum(axis=1)
        for r_sum in row_sums:
            if not within_tolerance(r_sum, 1):
                return False
        return True


def is_valid_chromosome(genes, n_states, n_symbols):
    start_probs = genes[:n_states]

    trans_probs_vector = genes[n_states:(n_states*(n_states + 1))]
    trans_probs_matrix = numpy.reshape(trans_probs_vector, (n_states, n_states))

    state_probs_vector = genes[n_states*(n_states + 1):]
    state_probs_matrix = numpy.reshape(state_probs_vector, (n_states, n_symbols))

    return (is_row_stochastic(start_probs)
    and is_row_stochastic(trans_probs_matrix)
    and is_row_stochastic(state_probs_matrix))


def test_random_chromosome():
    alphabet = list('abcdefghijklmn')
    chromosome = numpy.array(random_chromosome(6, alphabet))
    assert is_valid_chromosome(chromosome, 6, len(alphabet))

def test_single_row_cutpoint():
    alphabet = list('abcdefghijkl')
    crossover_func = single_row_cutpoint(6, len(alphabet))
    parents = numpy.array([random_chromosome(6, alphabet), random_chromosome(6, alphabet)])
    
    children = crossover_func(parents, (4, 0), 'soos')

    for child in children:
        assert is_valid_chromosome(child, 6, len(alphabet))