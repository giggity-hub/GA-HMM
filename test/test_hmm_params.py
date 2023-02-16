import pytest
from hmm.params import uniform_rand_stochastic_array
from test.assertions import assert_is_stochastic_across_axis


@pytest.mark.parametrize('shape', [(7, ), (3, 5), (13, 43, 17)])
def test_uniform_rand_stochastic_array(shape):
    stochastic_array = uniform_rand_stochastic_array(*shape)

    assert stochastic_array.shape == shape
    last_axis = len(shape) - 1
    assert_is_stochastic_across_axis(stochastic_array, axis=last_axis)