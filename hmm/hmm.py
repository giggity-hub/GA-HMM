from lib.utils import rand_stochastic_matrix
from typing import Tuple, List, Union, Callable, Annotated
from pomegranate import DiscreteDistribution, HiddenMarkovModel
import lib.utils as utils
import numpy
from numba import njit
from hmm.types import HmmParams, MultipleObservationSequences


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
def random_left_right_hmm_params(n_states: int, n_symbols: int) -> HmmParams:
    start_probs = numpy.zeros(n_states)
    start_probs[0] = 1

    emission_matrix = rand_stochastic_matrix(n_states, n_symbols)
    transition_matrix = left_right_transmat(n_states)

    return HmmParams(start_probs, emission_matrix, transition_matrix)


def multiple_observation_sequences_from_ndarray_list(ndarray_list: List[numpy.ndarray]) -> MultipleObservationSequences:
    """Concatenates a list of 1-D arrays and also remembers their slices

    Args:
        ndarray_list (List[numpy.ndarray]): _description_

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The first return Value is the concatenated array. The second value
        contains the slices so that ndarray_list[i] == unified_array[indices[i]:indices[i+1]]
    """
    indices = numpy.zeros((len(ndarray_list) + 1), dtype=int)

    for i in range(len(ndarray_list)):
        indices[i + 1] = indices[i] + len(ndarray_list[i])
    
    unified_array = numpy.concatenate(ndarray_list)
    
    return MultipleObservationSequences(slices=indices, arrays=unified_array, length=len(ndarray_list))