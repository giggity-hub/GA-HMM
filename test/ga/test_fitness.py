from ga.types import FitnessFunction
from ga.fitness import numba_mean_log_prob_fitness
import pytest
from ga.numba_ga import GaHMM
import numpy
from hmm.bw_numba import multiple_observation_sequences_from_ndarray_list
from hmm.hmm import random_left_right_hmm_params2
from test.assertions import assert_is_log_prob

@pytest.fixture
def gabw_mock():
    return GaHMM(
        n_symbols=128,
        n_states=4,
        population_size=13,
        n_generations=10
    )

@pytest.fixture
def observation_sequences(gabw_mock):
    sequences = []
    n_sequences = 10
    min_seq_length = 15
    max_seq_length = 500

    for i in range(n_sequences):
        seq_length = numpy.random.randint(low=min_seq_length, high=max_seq_length)
        rand_sequence = numpy.random.randint(low=0, high=gabw_mock.n_symbols, size=seq_length)
        sequences.append(rand_sequence)

    res = multiple_observation_sequences_from_ndarray_list(sequences)
    return res

@pytest.fixture
def fitness_func(observation_sequences):
    return numba_mean_log_prob_fitness(observation_sequences)

@pytest.fixture
def hmm_params(gabw_mock: GaHMM):
    return random_left_right_hmm_params2(gabw_mock.n_states, gabw_mock.n_symbols)


def test_is_log_prob(hmm_params, fitness_func: FitnessFunction):
    log_prob = fitness_func(hmm_params)
    assert_is_log_prob(log_prob)