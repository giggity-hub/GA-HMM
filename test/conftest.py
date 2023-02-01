import pytest
from data.digits import load_dataset
import hmm.bw_core as bw

import numpy
import math
import os
from numba import jit, config




@pytest.fixture
def numba_disable_jit():
    # os.environ["NUMBA_DISABLE_JIT"] = "1"
    config.DISABLE_JIT = True
    yield
    config.DISABLE_JIT = False
    # os.environ["NUMBA_DISABLE_JIT"] = "0"

@pytest.fixture
def numba_developer_mode():
    os.environ["NUMBA_DEVELOPER_MODE"] = "1"
    yield
    os.environ["NUMBA_DEVELOPER_MODE"] = "0"


def assert_is_row_stochastic(matrix: numpy.ndarray):
    max_deviation = 1e-8
    matrix = numpy.atleast_2d(matrix) #For the case that a vector with only 1-Dimension is supplied
    assert numpy.sum(matrix, axis=1) == pytest.approx(numpy.ones(len(matrix)))
    assert numpy.min(matrix) >= 0
    # max <= approx(1)
    assert numpy.max(matrix) < (1 + max_deviation)

def assert_is_log_prob(log_prob):
    assert not math.isnan(log_prob)
    assert log_prob < 0

@pytest.fixture
def digit_hmm_n_states():
    return 4

@pytest.fixture
def digit_hmm_n_symbols():
    return 128

@pytest.fixture
def digit_observation_sequences():
    training_data = load_dataset(dataset='train')
    digit = 0
    n_samples = 12
    samples = training_data[digit][:n_samples]

    observation_sequences = bw.multiple_observation_sequences_from_ndarray_list(samples)
    return observation_sequences

# @pytest.fixture
# def gabw(digit_observation_sequences):

#     mutation_func = numba_constant_uniform_mutation(0)
#     crossover_func = numba_single_point_crossover
#     parent_select_func = rank_selection
#     fitness_func = numba_mean_log_prob_fitness(digit_observation_sequences)

#     gabw = ga.GaHMM(
#         n_symbols=128,
#         n_states=4,
#         population_size=20,
#         n_generations=10,
#         fitness_func=fitness_func,
#         parent_select_func=parent_select_func,
#         mutation_func=mutation_func,
#         crossover_func=crossover_func,
#         keep_elitism=1
#     )
#     return gabw

# @pytest.fixture
# def digit_hmm_params_generator(digit_hmm_n_states, digit_hmm_n_symbols):
#     def generator():
#         while True:
#             start_vector, emission_matrix, transition_matrix = random_left_right_hmm_params(digit_hmm_n_states, digit_hmm_n_symbols)

#             hmm_params = bw.HmmParams(start_vector, emission_matrix, transition_matrix)
#             yield hmm_params
#     return generator

# def digit_hmm_chromosome_generator(digit_hmm_params_generator):
#     def generator():
#         while True:
#             hmm_params = digit_hmm_params_generator.next()
#             ga.GaHMM.

#     return generator

# @pytest.fixture
# def digit_hmm_slices(digit_hmm_n_states: int, digit_hmm_n_symbols: int):
#     slices = ga.GaHMM.calculate_slices(digit_hmm_n_states, digit_hmm_n_symbols)
#     return slices

