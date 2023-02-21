import math
import numpy
import pytest
from ga.numba_ga import GaHMM
from hmm.types import HmmParams, MultipleHmmParams

def assert_is_log_prob(log_prob):
    assert isinstance(log_prob, (numpy.floating, float))
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



def assert_is_stochastic_across_axis(ndarr: numpy.ndarray, axis):
    max_deviation = 1e-8
    sum_arr =  numpy.sum(ndarr, axis=axis)
    assert sum_arr == pytest.approx(numpy.ones_like(sum_arr))
    assert numpy.min(ndarr) >= 0
    assert numpy.max(ndarr) < (1 + max_deviation)

def assert_is_row_stochastic(ndarr: numpy.ndarray):
    max_deviation = 1e-8
    last_axis = ndarr.ndim -1
    assert_is_stochastic_across_axis(ndarr, axis=last_axis)

# def assert_chromosomes_are_row_stochastic(chromosomes: numpy.ndarray, gabw: GaHMM):
#     for i in range(len(chromosomes)):
#         hmm_params = gabw.chromosome2hmm_params(chromosomes[i])
#         assert_is_row_stochastic(hmm_params.start_vector)
#         assert_is_row_stochastic(hmm_params.emission_matrix)
#         assert_is_row_stochastic(hmm_params.transition_matrix)


def assert_valid_hmm_params(hmm_params: HmmParams ):
    start_vector, emission_matrix, transition_matrix = hmm_params
    n_states, n_symbols = emission_matrix.shape

    assert start_vector.shape == (n_states, )
    assert transition_matrix.shape == (n_states, n_states)

    assert_is_row_stochastic(start_vector)
    assert_is_row_stochastic(transition_matrix)
    assert_is_row_stochastic(emission_matrix)




def assert_valid_multiple_hmm_params(hmm_params: MultipleHmmParams):
    PIs, Bs, As = hmm_params

    n_hmms, n_states, n_symbols = Bs.shape

    assert PIs.shape == (n_hmms, n_states)
    assert Bs.shape == (n_hmms, n_states, n_symbols)
    assert As.shape == (n_hmms, n_states, n_states)

    assert_is_stochastic_across_axis(PIs, axis=1)
    assert_is_stochastic_across_axis(Bs, axis=2 )
    assert_is_stochastic_across_axis(As, axis=2)
    



def assert_no_shared_memory(ndarray: numpy.ndarray):
    # If an Arrays shares memory with another array the base points to the array it shares memory with
    # If an Array does not share memory the base is None
    assert not type(ndarray.base) == numpy.ndarray
    assert ndarray.base == None

def assert_hmm_params_are_within_tolerance(hmm_params_1: HmmParams, hmm_params_2: HmmParams, atol):
    assert numpy.allclose(hmm_params_1.PI, hmm_params_2.PI, atol=atol)
    assert numpy.allclose(hmm_params_1.B, hmm_params_2.B, atol=atol)
    assert numpy.allclose(hmm_params_1.A, hmm_params_2.A, atol=atol)


def assert_multiple_hmm_params_are_equal(hmm_params_1: MultipleHmmParams, hmm_params_2: MultipleHmmParams):
    
    assert numpy.array_equal(hmm_params_1.PIs, hmm_params_2.PIs)
    assert numpy.array_equal(hmm_params_1.Bs, hmm_params_2.Bs)
    assert numpy.array_equal(hmm_params_1.As, hmm_params_2.As)


def assert_no_zeros_in_emissions_of_multiple_hmm_params(hmm_params: MultipleHmmParams):
    assert numpy.count_nonzero(hmm_params.Bs) == hmm_params.Bs.size

