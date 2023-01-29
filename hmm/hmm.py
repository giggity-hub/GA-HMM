from lib.utils import rand_stochastic_matrix
from typing import Tuple, List, Union, Callable, Annotated
from pomegranate import DiscreteDistribution, HiddenMarkovModel
import lib.utils as utils
import numpy
from numba import njit
from hmm.types import HmmParams


ParamGeneratorFunction = Callable[
    [
        Annotated[int, 'n_states'],
        Annotated[int, 'n_symbols']
    ],
    Tuple[numpy.array, numpy.ndarray, numpy.ndarray]
]


ParamGeneratorFunction2 = Callable[
    [
        Annotated[int, 'n_states'],
        Annotated[int, 'n_symbols']
    ],
    HmmParams
]

@njit(inline='always')
def left_right_transmat(n_states: int) -> numpy.ndarray:
    matrix = numpy.zeros(shape=(n_states, n_states))

    for s in range(n_states -1):
        matrix[s, s] = 0.5
        matrix[s, s+1] = 0.5
    
    matrix[-1, -1] = 1
    return matrix

@njit
def random_left_right_hmm_params(n_states: int, n_symbols: int) -> Tuple[numpy.array, numpy.ndarray, numpy.ndarray]:
    start_probs = numpy.zeros(n_states)
    start_probs[0] = 1

    emission_probs = rand_stochastic_matrix(n_states, n_symbols)
    transition_probs = left_right_transmat(n_states)

    return start_probs, emission_probs, transition_probs

def random_left_right_hmm_params2(n_states: int, n_symbols: int) -> HmmParams:
    start_probs = numpy.zeros(n_states)
    start_probs[0] = 1

    emission_matrix = rand_stochastic_matrix(n_states, n_symbols)
    transition_matrix = left_right_transmat(n_states)

    return HmmParams(start_probs, emission_matrix, transition_matrix)

def get_discrete_hmm_params(hmm: HiddenMarkovModel) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        # Step 1: Get that motherfucking transitions shit
    transition_matrix: numpy.matrix = hmm.dense_transition_matrix()

    # The Start Probs are the outgoing transitions from the silent start state
    pi = transition_matrix[hmm.start_index, :hmm.silent_start]
    pi = numpy.array(pi)

    # Remove Silent States
    a = transition_matrix[:hmm.silent_start, :hmm.silent_start]
    a = numpy.array(a)


    n_symbols = len(hmm.states[0].distribution)
    n_states = hmm.silent_start

    b = numpy.empty(shape=(n_states, n_symbols))

    for i in range(n_states):
        state = hmm.states[i]
        # Reminder: We don't check whether the values are in the correct order (they better be otherwise this won't work)
        b[i: ] = state.distribution.values()

    return (pi, b, a)