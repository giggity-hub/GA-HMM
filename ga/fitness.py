# fitness function 1
# fitness with bound fixed length of input
from ga.ga import Chromosome, GaHMM
from typing import List, Callable
from hmm.bw import BaumWelch
from hmm.bw_numba import calc_mean_log_prob, HmmParams
import numpy
from numba import jit, njit


def mean_log_prob_fitness(samples: List[List[int]]) -> Callable[[Chromosome, GaHMM], float]:
    """_summary_

    Args:
        samples should be a list of observations

    Returns:
        callable[[Individual, GeneticAlgorithm], float]: _description_
    """
    
    def fitness_func(chromosome: Chromosome, gabw: GaHMM) -> float:
        s = chromosome.start_vector
        e = chromosome.emission_matrix
        t = chromosome.transition_matrix
        bw = BaumWelch(s, e, t)
        
        mean_log_prob = bw.calc_mean_log_prob(samples)
        return mean_log_prob
    

    return fitness_func


def numba_mean_log_prob_fitness(samples: List[List[int]]) -> Callable[[HmmParams], float]:
    
    def fitness_func(hmm_params: HmmParams) -> float:
        
        mean_log_prob = calc_mean_log_prob(hmm_params, samples)
        return mean_log_prob
    

    return fitness_func


# def log_prob_fitnessOLD(samples: List[List[int]]) -> Callable[[Individual, GeneticAlgorithm], float]:
#     """_summary_

#     Args:
#         samples should be a list of observations

#     Returns:
#         callable[[Individual, GeneticAlgorithm], float]: _description_
#     """
    

#     def fitness_func(individual: Individual, ga_instance: GeneticAlgorithm=None) -> float:

#         child_hmm = individual.to_hmm()

#         total_score = 0
#         for sample in samples:
#             total_score += child_hmm.log_probability(sample)
        
#         mean_score = total_score / len(samples)
#         return mean_score
    

#     return fitness_func