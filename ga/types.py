import numpy.typing as npt
import numpy
from typing import NamedTuple, Callable, Annotated, TYPE_CHECKING
from hmm.types import HmmParams


if TYPE_CHECKING:
    from ga.numba_ga import GaHMM


class ChromosomeFields(NamedTuple):
    N_STATES: int
    N_SYMBOLS: int
    FITNESS: int
    RANK: int

ChromosomeMask = npt.NDArray[numpy.bool_]

class ChromosomeSlices(NamedTuple):
    PI: slice
    B: slice
    A: slice

Chromosome = npt.NDArray[numpy.float64]

Population = npt.NDArray[numpy.float64]

class RangeTuple(NamedTuple):
    start: int
    stop: int
    step: int

class ChromosomeRanges(NamedTuple):
    PI: RangeTuple
    B: RangeTuple
    A: RangeTuple



# class ChromosomeSlices(NamedTuple):
#     start_probs: SliceTuple
#     emission_probs: SliceTuple
#     transition_probs: SliceTuple
#     fitness: SliceTuple
#     rank: SliceTuple





FitnessFunction = Callable[[HmmParams], float]

CrossoverFunction = Callable[
    [
        Annotated[numpy.ndarray, 'parents'], 
        Annotated[ChromosomeSlices, 'slices'],
        'GaHMM'
    ],
    numpy.ndarray
]

MutationFunction = Callable[
    [
        Annotated[numpy.ndarray, 'chromosome'],
        'GaHMM'
    ],
    Annotated[numpy.ndarray, 'chromosome']
]

SelectionFunction = Callable[
    [
        Annotated[numpy.ndarray, 'population' ],
        Annotated[int, 'n_offspring'],
        ChromosomeSlices,
        'GaHMM'
    ],
    Annotated[numpy.ndarray, 'parents']
]