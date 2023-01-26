# fitness function 1
# fitness with bound fixed length of input
# from ga.ga import Chromosome, GaHMM
# from typing import List, Callable
# from hmm.bw import BaumWelch
from hmm.bw_numba import calc_mean_log_prob
from hmm.types import HmmParams, MultipleObservationSequences
from numba import jit, njit
from ga.types import FitnessFunction



def numba_mean_log_prob_fitness(samples: MultipleObservationSequences) -> FitnessFunction:
    
    def fitness_func(hmm_params: HmmParams) -> float:
        
        mean_log_prob = calc_mean_log_prob(hmm_params, samples)
        # print(mean_log_prob)
        # print(hmm_params)
        return mean_log_prob
    
    
    return fitness_func
