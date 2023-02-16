from lib.utils import uniform_rand_stochastic_matrix, conditional_rand_stochastic_matrix
from typing import Tuple, List, Union, Callable, Annotated
from pomegranate import DiscreteDistribution, HiddenMarkovModel
import lib.utils as utils
import numpy
from numba import njit
from hmm.types import HmmParams, MultipleObservationSequences, MultipleHmmParams
rng = numpy.random.default_rng()


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
def uniform_left_right_transmat(n_states: int) -> numpy.ndarray:
    matrix = numpy.zeros(shape=(n_states, n_states))

    for s in range(n_states -1):
        matrix[s, s] = 0.5
        matrix[s, s+1] = 0.5
    
    matrix[-1, -1] = 1
    return matrix


@njit
def uniform_random_left_right_hmm_params(n_states: int, n_symbols: int) -> HmmParams:
    start_probs = numpy.zeros(n_states)
    start_probs[0] = 1

    emission_matrix = uniform_rand_stochastic_matrix(n_states, n_symbols)
    transition_matrix = uniform_left_right_transmat(n_states)

    return HmmParams(start_probs, emission_matrix, transition_matrix)


def conditional_random_ergodic_hmm_params(n_states: int, n_symbols: int) -> HmmParams:
    start_probs = conditional_rand_stochastic_matrix(1, n_states).flatten()
    emission_matrix = conditional_rand_stochastic_matrix(n_states, n_symbols)
    transition_matrix = conditional_rand_stochastic_matrix(n_states, n_states)

    return HmmParams(start_probs, emission_matrix, transition_matrix)



def uniform_rand_stochastic_array(*size) -> numpy.array:
    arr = rng.random(size)
    sums_of_highest_dim = numpy.sum(arr, axis=-1, keepdims=True)
    normalized_arr = arr / sums_of_highest_dim
    return normalized_arr

def _uniform_start_vector(length: int, start_states) -> numpy.array:
    start_probs = numpy.zeros(length)
    start_probs[start_states] = 1 / length
    return start_probs

def multiple_uniform_random_left_right_hmm_params(n_states, n_symbols, n_hmms):

    start_vector = _uniform_start_vector(length=n_states, start_states=[0])
    start_vectors = numpy.tile((n_hmms, 1), start_vector)

    emission_matrices = uniform_rand_stochastic_array(n_hmms, n_states, n_symbols)

    transition_matrix = uniform_left_right_transmat(n_states)
    transition_matrices = numpy.tile((n_hmms, 1, 1), transition_matrix)

    return MultipleHmmParams(start_vectors, emission_matrices, transition_matrices)





















# def multiple_observation_sequences_from_ndarray_list(ndarray_list: List[numpy.ndarray]) -> MultipleObservationSequences:
#     """Concatenates a list of 1-D arrays and also remembers their slices

#     Args:
#         ndarray_list (List[numpy.ndarray]): _description_

#     Returns:
#         Tuple[numpy.ndarray, numpy.ndarray]: The first return Value is the concatenated array. The second value
#         contains the slices so that ndarray_list[i] == unified_array[indices[i]:indices[i+1]]
#     """
#     indices = numpy.zeros((len(ndarray_list) + 1), dtype=int)

#     for i in range(len(ndarray_list)):
#         indices[i + 1] = indices[i] + len(ndarray_list[i])
    
#     unified_array = numpy.concatenate(ndarray_list)
    
#     return MultipleObservationSequences(slices=indices, arrays=unified_array, length=len(ndarray_list))