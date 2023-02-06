import numpy
from typing import NamedTuple

class MultipleObservationSequences(NamedTuple):
    """_summary_
    """
    slices: numpy.ndarray
    arrays: numpy.ndarray
    length: int

class HmmParams(NamedTuple):
    start_vector: numpy.ndarray
    emission_matrix: numpy.ndarray
    transition_matrix: numpy.ndarray


class MultipleHmmParams(NamedTuple):
    PIs: numpy.ndarray
    Bs: numpy.ndarray
    As: numpy.ndarray