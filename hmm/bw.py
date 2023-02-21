from hmm.bw_core import reestimate_multiple_observations, reestimate_single_observation, calc_log_prob
import numpy
from numba import njit
from hmm.types import HmmParams, MultipleObservationSequences, MultipleHmmParams
from typing import Tuple
import numpy
import math



@njit(cache=True)
def train_single_hmm(hmm_params: HmmParams, all_observations: MultipleObservationSequences, n_iterations: int=1):
    pi, b, a = hmm_params

    total_log_prob_after_iteration = numpy.empty(n_iterations)
    for i in range(n_iterations):
        pi, b, a, log_prob_total = reestimate_multiple_observations(pi, b, a, all_observations )
        if math.isnan(log_prob_total) and i > 0:
            total_log_prob_after_iteration[i:] = total_log_prob_after_iteration[i-1]
            break
        else:
            total_log_prob_after_iteration[i] = log_prob_total

    reestimated_hmm_params = HmmParams(pi, b, a)
    return reestimated_hmm_params, total_log_prob_after_iteration

# @njit
# def train_single_observation(hmm_params: HmmParams, all_observations: numpy.ndarray, n_iterations: int=1):
#     pi, b, a = hmm_params

#     total_log_prob_after_iteration = numpy.empty(n_iterations)
#     for i in range(n_iterations):
#         pi, b, a, log_prob_total = reestimate_single_observation(pi, b, a, all_observations )
#         total_log_prob_after_iteration[i] = log_prob_total

#     reestimated_hmm_params = HmmParams(pi, b, a)
#     return reestimated_hmm_params, total_log_prob_after_iteration

@njit(cache=True)
def train_single_hmm_with_stop_conditions(
    hmm_params: HmmParams, 
    all_observations: MultipleObservationSequences,
    max_n_iterations: int,
    stop_if_improvement_less_than: float = 0.0001,
    stop_if_probability_greater_than: float = 0.9999,
    ):

    max_log_prob = numpy.log(stop_if_probability_greater_than)
    min_log_prob_improvement = numpy.log(stop_if_improvement_less_than)

    pi, b, a = hmm_params

    total_log_prob_after_iteration = numpy.empty(max_n_iterations)
    i = 0
    while (i < max_n_iterations):
        pi, b, a, log_prob_total = reestimate_multiple_observations(pi, b, a, all_observations )
        total_log_prob_after_iteration[i] = log_prob_total
        
        improvement = float('inf') if i < 1 else (log_prob_total - total_log_prob_after_iteration[i-1])
        if (log_prob_total > max_log_prob) or (improvement < min_log_prob_improvement):
            break

        i+= 1

    did_bw_stop_before_max_iteration = i < max_n_iterations
    if did_bw_stop_before_max_iteration:
        total_log_prob_after_iteration[i:] = total_log_prob_after_iteration[i]
    
    
    reestimated_hmm_params = HmmParams(pi, b, a)
    return reestimated_hmm_params, total_log_prob_after_iteration



@njit(cache=True)
def calc_total_log_prob(hmm_params: HmmParams, all_observation_seqs: MultipleObservationSequences):
    pi, b, a = hmm_params
    total_log_prob = 0

    slices, array, length = all_observation_seqs

    for i in range(length):
        start = slices[i]
        stop = slices[i+1]

        single_observation_seq = array[start:stop]
        try:
            log_prob = calc_log_prob(pi, b, a, single_observation_seq)
            total_log_prob += log_prob
        except:
            return - numpy.inf
        

    return total_log_prob

@njit(cache=True)
def calc_mean_log_prob(hmm_params: HmmParams, all_observation_seqs: MultipleObservationSequences):
    total_log_prob = calc_total_log_prob(hmm_params, all_observation_seqs)
    mean_log_prob = total_log_prob / all_observation_seqs.length
    return mean_log_prob

@njit(cache=True)
def calc_total_log_prob_for_multiple_hmms(
    all_hmm_params: MultipleHmmParams,
    all_observation_seqs: MultipleObservationSequences
    ) -> numpy.ndarray:

    start_vectors, emission_matrices, transition_matrices = all_hmm_params
    n_hmms = start_vectors.shape[0]

    total_log_prob_for_hmm = numpy.empty(n_hmms)

    for hmm_index in range(n_hmms):
        pi = start_vectors[hmm_index]
        b = emission_matrices[hmm_index]
        a = transition_matrices[hmm_index]
        hmm_params = HmmParams(pi, b, a)

        total_log_prob = calc_total_log_prob(hmm_params, all_observation_seqs)
        total_log_prob_for_hmm[hmm_index] = total_log_prob

    return total_log_prob_for_hmm


@njit(cache=True)
def train_multiple_hmms(
    all_hmm_params: MultipleHmmParams, 
    all_observation_seqs: MultipleObservationSequences, 
    n_iterations: int
    ) -> Tuple[MultipleHmmParams, numpy.ndarray]:
    """hmm_log_prob_after_iteration[i, j] is the total log probability of the ith hmm after the jth bw iteration

    Args:
        all_hmm_params (MultipleHmmParams): _description_
        all_observation_seqs (MultipleObservationSequences): _description_
        n_iterations (int): _description_

    Returns:
        Tuple[MultipleHmmParams, numpy.ndarray]: _description_
    """

    start_vectors, emission_matrices, transition_matrices = all_hmm_params

    n_hmms = start_vectors.shape[0]
    hmm_log_prob_after_iteration = numpy.empty((n_hmms, n_iterations))

    for hmm_index in range(n_hmms):
        pi = start_vectors[hmm_index]
        b = emission_matrices[hmm_index]
        a = transition_matrices[hmm_index]

        for iteration_index in range(n_iterations):
            pi, b, a, log_prob_total = reestimate_multiple_observations(pi, b, a, all_observation_seqs)
            hmm_log_prob_after_iteration[hmm_index, iteration_index] = log_prob_total

        start_vectors[hmm_index] = pi
        emission_matrices[hmm_index] = b
        transition_matrices[hmm_index] = a

    reestimated_hmm_params = MultipleHmmParams(start_vectors, emission_matrices, transition_matrices)
    return reestimated_hmm_params, hmm_log_prob_after_iteration