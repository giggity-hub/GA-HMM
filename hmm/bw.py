from hmm.bw_core import reestimate_multiple_observations, reestimate_single_observation, calc_log_prob
import numpy
from numba import njit
from hmm.types import HmmParams, MultipleObservationSequences

def train_multiple_observations(hmm_params: HmmParams, all_observations: MultipleObservationSequences, n_iterations: int=1):
    pi, b, a = hmm_params

    total_log_prob_after_iteration = numpy.empty(n_iterations)
    for i in range(n_iterations):
        pi, b, a, log_prob_total = reestimate_multiple_observations(pi, b, a, all_observations )
        total_log_prob_after_iteration[i] = log_prob_total

    reestimated_hmm_params = HmmParams(pi, b, a)
    return reestimated_hmm_params, total_log_prob_after_iteration

def train_single_observation(hmm_params: HmmParams, all_observations: numpy.ndarray, n_iterations: int=1):
    pi, b, a = hmm_params

    total_log_prob_after_iteration = numpy.empty(n_iterations)
    for i in range(n_iterations):
        pi, b, a, log_prob_total = reestimate_single_observation(pi, b, a, all_observations )
        total_log_prob_after_iteration[i] = log_prob_total

    reestimated_hmm_params = HmmParams(pi, b, a)
    return reestimated_hmm_params, total_log_prob_after_iteration


@njit(inline='always')
def calc_total_log_prob(hmm_params: HmmParams, all_observation_seqs: MultipleObservationSequences):
    pi, b, a = hmm_params
    total_log_prob = 0

    slices, array, length = all_observation_seqs

    for i in range(length):
        start = slices[i]
        stop = slices[i+1]

        single_observation_seq = array[start:stop]
        log_prob = calc_log_prob(pi, b, a, single_observation_seq)
        total_log_prob += log_prob

    return total_log_prob


def calc_mean_log_prob(hmm_params: HmmParams, all_observation_seqs: MultipleObservationSequences):
    total_log_prob = calc_total_log_prob(hmm_params, all_observation_seqs)
    mean_log_prob = total_log_prob / all_observation_seqs.length
    return mean_log_prob