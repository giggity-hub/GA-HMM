import pytest 
from pytest import approx
from hmm.hmm import random_left_right_hmm_params
import numpy


@pytest.mark.parametrize("n_states, n_symbols", [(5,5), (4, 16), (1,1)])
def test_random_left_right_hmm_params(n_states, n_symbols):

    start_vector, emission_matrix, transition_matrix = random_left_right_hmm_params(n_states, n_symbols)

    assert start_vector.shape == (n_states, )
    assert emission_matrix.shape == (n_states, n_symbols)
    assert transition_matrix.shape == (n_states, n_states)

    # Test row stochasticity
    assert numpy.sum(start_vector) == approx(1)
    assert numpy.min(start_vector) >= 0
    assert numpy.max(start_vector) <= 1

    assert numpy.sum(emission_matrix, axis=1) == approx(numpy.ones(n_states))
    assert numpy.min(emission_matrix) >= 0
    assert numpy.max(emission_matrix) <= 1

    assert numpy.sum(transition_matrix, axis=1) == approx(numpy.ones(n_states))
    assert numpy.min(transition_matrix) >= 0 
    assert numpy.max(transition_matrix) <= 1