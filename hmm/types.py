import numpy
from typing import NamedTuple

class MultipleObservationSequences(NamedTuple):
    """_summary_
    """
    slices: numpy.ndarray
    arrays: numpy.ndarray
    length: int

class HmmParams(NamedTuple):
    PI: numpy.ndarray
    B: numpy.ndarray
    A: numpy.ndarray


class MultipleHmmParams(NamedTuple):
    PIs: numpy.ndarray
    Bs: numpy.ndarray
    As: numpy.ndarray