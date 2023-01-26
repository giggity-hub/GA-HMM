import numpy.typing as npt
import numpy
from typing import NamedTuple, Callable, Annotated, TYPE_CHECKING

from hmm.types import HmmParams


if TYPE_CHECKING:
    from ga.numba_ga import GaHMM


ChromosomeMask = npt.NDArray[numpy.bool_]


class SliceTuple(NamedTuple):
    start: int
    stop: int
    step: int

class ChromosomeSlices(NamedTuple):
    start_probs: SliceTuple
    emission_probs: SliceTuple
    transition_probs: SliceTuple
    fitness: SliceTuple
    rank: SliceTuple

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
        ChromosomeSlices,
        ChromosomeMask,
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