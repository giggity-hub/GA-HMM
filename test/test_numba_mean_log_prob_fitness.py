from data.digits import load_dataset
from ga.fitness import numba_mean_log_prob_fitness
from hmm.hmm import random_left_right_hmm_params
from hmm.bw_numba import HmmParams
import math
from numba import jit, config



def setup_function(function):
    config.NUMBA_DEVELOPER_MODE = 1


def teardown_function(function):
    config.NUMBA_DEVELOPER_MODE = 0


def test_numba_mean_log_prob_fitness():
    training_data = load_dataset(dataset='train')
    digit = 0
    n_samples = 12
    samples = training_data[digit][:n_samples]
    fitness_func = numba_mean_log_prob_fitness(samples)

    n_states = 5
    n_symbols = 16
    start_vector, emission_matrix, transition_matrix = random_left_right_hmm_params(n_states, n_symbols)

    hmm_params = HmmParams(start_vector, emission_matrix, transition_matrix)

    log_prob = fitness_func(hmm_params)

    assert not math.isnan(log_prob)
    assert log_prob < 0


