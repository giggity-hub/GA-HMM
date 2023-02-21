import pytest
from hmm.params import create_multiple_uniform_random_left_right_hmm_params, uniform_random_left_right_hmm_params
from hmm.types import MultipleHmmParams, HmmParams
import ga.representation as representation

from test.assertions import (
    assert_multiple_hmm_params_are_equal,
    assert_valid_hmm_params,
    assert_valid_multiple_hmm_params
)


@pytest.fixture
def multiple_hmm_params():
    hmm_params = create_multiple_uniform_random_left_right_hmm_params(n_hmms=23, n_states=17, n_symbols=37)
    return hmm_params


@pytest.fixture
def hmm_params():
    hmm_params = uniform_random_left_right_hmm_params(n_states=13, n_symbols=59)
    return hmm_params

def test_convert_between_multiple_hmm_params_and_population(multiple_hmm_params: MultipleHmmParams):
    n_hmms, n_states, n_symbols = multiple_hmm_params.Bs.shape

    chromosomes = representation.multiple_hmm_params_as_population(multiple_hmm_params)
    new_multiple_hmm_params = representation.population_as_multiple_hmm_params(chromosomes)

    assert_multiple_hmm_params_are_equal(multiple_hmm_params, new_multiple_hmm_params)



def test_convert_between_hmm_params_and_chromosome(hmm_params: HmmParams):
    chromosome = representation.hmm_params_as_chromosome(hmm_params)
    n_states, n_symbols = hmm_params.B.shape
    new_hmm_params = representation.chromosome_as_hmm_params(chromosome)

    assert_valid_hmm_params(new_hmm_params)



def test_calc_chromosome_ranges():
    n_states = 13
    n_symbols = 43
    ranges = representation.calc_chromosome_ranges(n_states, n_symbols)

    assert ranges.PI.start == 0
    assert ranges.PI.stop == n_states
    assert ranges.PI.step == n_states

@pytest.fixture
def hmm_params_list():
    n_states = 13
    n_symbols = 17
    n_hmms = 7
    params = [uniform_random_left_right_hmm_params(n_states, n_symbols) for i in range(n_hmms)]
    return params

def test_hmm_params_list_as_multiple_hmm_params(hmm_params_list):
    multiple_hmm_params = representation.hmm_params_list_as_multiple_hmm_params(hmm_params_list)

    assert_valid_multiple_hmm_params(multiple_hmm_params)