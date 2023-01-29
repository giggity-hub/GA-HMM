from data.digits import load_dataset
from ga.fitness import numba_mean_log_prob_fitness
from hmm.hmm import random_left_right_hmm_params
from hmm.bw_numba import HmmParams, multiple_observation_sequences_from_ndarray_list
import math
from numba import jit, config
import pytest



# def setup_function(function):
#     config.NUMBA_DEVELOPER_MODE = 1


# def teardown_function(function):
#     config.NUMBA_DEVELOPER_MODE = 0


@pytest.fixture
def samples():
    """These samples only work with an 128 Observation Symbol-HMM

    Returns:
        _type_: _description_
    """
    training_data = load_dataset(dataset='train')
    digit = 0
    n_samples = 12
    samples = training_data[digit][:n_samples]
    return samples

@pytest.fixture
def hmm_params():
    n_states = 5
    n_symbols = 128
    start_vector, emission_matrix, transition_matrix = random_left_right_hmm_params(n_states, n_symbols)

    hmm_params = HmmParams(start_vector, emission_matrix, transition_matrix)
    return hmm_params


def test_numba_mean_log_prob_fitness(samples, hmm_params):

    samples = multiple_observation_sequences_from_ndarray_list(samples)
    fitness_func = numba_mean_log_prob_fitness(samples)

    log_prob = fitness_func(hmm_params)

    assert not math.isnan(log_prob)
    assert log_prob < 0


