import pytest
from hmm.params import uniform_rand_stochastic_array, multiple_uniform_random_left_right_hmm_params
from test.assertions import assert_is_stochastic_across_axis, assert_valid_multiple_hmm_params


@pytest.mark.parametrize('shape', [(7, ), (3, 5), (13, 43, 17)])
def test_uniform_rand_stochastic_array(shape):
    stochastic_array = uniform_rand_stochastic_array(*shape)

    assert stochastic_array.shape == shape
    last_axis = len(shape) - 1
    assert_is_stochastic_across_axis(stochastic_array, axis=last_axis)



def test_multiple_uniform_random_left_right_hmm_params():
    n_hmms = 13
    n_states = 17
    n_symbols = 23

    hmm_params = multiple_uniform_random_left_right_hmm_params(n_hmms, n_states, n_symbols)
    PIs, Bs, As = hmm_params

    assert PIs.shape == (n_hmms, n_states)
    assert Bs.shape == (n_hmms, n_states, n_symbols)
    assert As.shape == (n_hmms, n_states, n_states)

    assert_valid_multiple_hmm_params(hmm_params)
