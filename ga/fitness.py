# fitness function 1
# fitness with bound fixed length of input
# from ga.ga import Chromosome, GaHMM
# from typing import List, Callable
# from hmm.bw import BaumWelch
from hmm.bw import calc_mean_log_prob
from hmm.types import HmmParams, MultipleObservationSequences
from numba import jit, njit
from ga.types import FitnessFunction
import numpy



