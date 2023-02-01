import math
import numpy
import pytest
from ga.numba_ga import GaHMM
from hmm.types import HmmParams

def assert_is_log_prob(log_prob):
    assert type(log_prob) == numpy.float64
    assert not math.isnan(log_prob)
    assert log_prob < 0

def assert_no_nan_values(ndarr):
    assert not numpy.any(numpy.isnan(ndarr))

def assert_all_values_leq_one(ndarr):
    assert numpy.all(numpy.less_equal(ndarr, 1))

def assert_all_values_geq_zero(ndarr):
    assert numpy.all(numpy.greater_equal(ndarr, 0))


def assert_all_values_are_probabilities(ndarr):
    assert_no_nan_values(ndarr)
    assert_all_values_geq_zero(ndarr)
    assert_all_values_leq_one(ndarr)

def assert_all_values_lt_zero(ndarr):
    assert numpy.all(numpy.less(ndarr, 0))

def assert_all_values_are_log_probabilities(ndarr):
    assert_no_nan_values(ndarr)
    assert_all_values_lt_zero(ndarr)

def assert_is_row_stochastic(matrix: numpy.ndarray):
    max_deviation = 1e-8
    matrix = numpy.atleast_2d(matrix) #For the case that a vector with only 1-Dimension is supplied
    assert numpy.sum(matrix, axis=1) == pytest.approx(numpy.ones(len(matrix)))
    assert numpy.min(matrix) >= 0
    assert numpy.max(matrix) < (1 + max_deviation)

def assert_chromosomes_are_row_stochastic(chromosomes: numpy.ndarray, gabw: GaHMM):
    for i in range(len(chromosomes)):
        hmm_params = gabw.chromosome2hmm_params(chromosomes[i])
        assert_is_row_stochastic(hmm_params.start_vector)
        assert_is_row_stochastic(hmm_params.emission_matrix)
        assert_is_row_stochastic(hmm_params.transition_matrix)


def assert_valid_hmm_params(hmm_params: HmmParams ):
    start_vector, emission_matrix, transition_matrix = hmm_params
    n_states, n_symbols = emission_matrix.shape

    assert start_vector.shape == (n_states, )
    assert transition_matrix.shape == (n_states, n_states)

    assert_is_row_stochastic(start_vector)
    assert_is_row_stochastic(transition_matrix)
    assert_is_row_stochastic(emission_matrix)

    

def assert_no_shared_memory(ndarray: numpy.ndarray):
    # If an Arrays shares memory with another array the base points to the array it shares memory with
    # If an Array does not share memory the base is None
    assert not type(ndarray.base) == numpy.ndarray
    assert ndarray.base == None

def assert_hmm_params_are_equal(hmm_params_1: HmmParams, hmm_params_2: HmmParams, atol):
    assert numpy.allclose(hmm_params_1.start_vector, hmm_params_2.start_vector, atol=atol)
    assert numpy.allclose(hmm_params_1.emission_matrix, hmm_params_2.emission_matrix, atol=atol)
    assert numpy.allclose(hmm_params_1.transition_matrix, hmm_params_2.transition_matrix, atol=atol)