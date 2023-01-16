from lib.utils import rand_stochastic_matrix
from typing import Tuple, List, Union, Callable
from pomegranate import DiscreteDistribution, HiddenMarkovModel
import lib.utils as utils
import numpy

def left_right_transmat(n_states: int) -> numpy.ndarray:
    matrix = numpy.ndarray(shape=(n_states, n_states), dtype=float)

    for s in range(n_states -1):
        matrix[s, s] = 0.5
        matrix[s, s+1] = 0.5
    
    matrix[-1, -1] = 1
    return matrix

def random_left_right_hmm_params(n_states: int, n_symbols: int) -> Tuple[numpy.array, numpy.ndarray, numpy.ndarray]:
    start_probs = numpy.zeros(n_states)
    start_probs[0] = 1

    emission_probs = rand_stochastic_matrix(n_states, n_symbols)
    transition_probs = left_right_transmat(n_states)

    return start_probs, emission_probs, transition_probs

