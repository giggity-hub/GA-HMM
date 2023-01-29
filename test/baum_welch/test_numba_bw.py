from hmm.bw_numba import  HmmParams, multiple_observation_sequences_from_ndarray_list
import pytest
import csv
import pandas as pd
import numpy
from typing import List, NamedTuple
import hmm.bw_numba as bw
import math
from numba import jit, config
from hmm.hmm import random_left_right_hmm_params



def setup_function(function):
    config.NUMBA_DEVELOPER_MODE = 1


def teardown_function(function):
    config.NUMBA_DEVELOPER_MODE = 0


def first_50k_observations_in_brown_corpus():
    stamp_observation_sequence = []
    observation_symbols = list('abcdefghijklmnopqrstuwxyz ')

    with open('test/baum_welch/brown.txt') as f:

        while len(stamp_observation_sequence) < 50000:
            character = f.read(1).lower()
            
            if character in observation_symbols:
                stamp_observation_sequence.append(character)
    
    return stamp_observation_sequence

def convert_observation_symbols_to_integer_value(str_observation_sequence: List[str]) -> numpy.ndarray:
    n_observations = len(str_observation_sequence)
    int_observation_sequence = numpy.ndarray(n_observations, dtype=int)

    int_values = list('abcdefghijklmnopqrstuvwxyz ')
    n_observation_symbols = 27
    str_values = [i for i in range(n_observation_symbols)]
    character_int_value = dict(zip(int_values, str_values))

    for i in range(n_observations):
        character = str_observation_sequence[i]
        int_observation_sequence[i] = character_int_value[character]
    
    return int_observation_sequence

@pytest.fixture
def stamp_observation_sequence():
    str_observation_sequence = first_50k_observations_in_brown_corpus()
    int_observation_sequence = convert_observation_symbols_to_integer_value(str_observation_sequence)

    return int_observation_sequence


@pytest.fixture
def data_frames():
    filepath_Pi = 'test/baum_welch/stamp_Pi.csv'
    filepath_B = 'test/baum_welch/stamp_B.csv'
    filepath_A = 'test/baum_welch/stamp_A.csv'

    df_Pi = pd.read_csv(filepath_Pi, sep=' ')
    df_A = pd.read_csv(filepath_A ,sep=' ')
    df_B = pd.read_csv(filepath_B, sep=' ')

    return (df_Pi, df_B, df_A)


@pytest.fixture
def stamp_initial_hmm_params(data_frames):
    df_Pi, df_B, df_A = data_frames

    initial_start_vector = df_Pi.loc[:, 'initial_s1':'initial_s2'].to_numpy().flatten()
    initial_emission_matrix = df_B.loc[:, 'initial_s1':'initial_s2'].to_numpy().T
    initial_transition_matrix = df_A.loc[:, 'initial_s1':'initial_s2'].to_numpy().T

    return HmmParams(
        initial_start_vector,
        initial_emission_matrix,
        initial_transition_matrix
    )
    
@pytest.fixture
def stamp_final_hmm_params(data_frames):
    df_Pi, df_B, df_A = data_frames

    final_start_vector = df_Pi.loc[:, 'final_s1': 'final_s2'].to_numpy().flatten()
    final_emission_matrix = df_B.loc[:, 'final_s1': 'final_s2'].to_numpy().T
    final_transition_matrix = df_A.loc[:, 'final_s1':'final_s2'].to_numpy().T 

    return HmmParams(
        final_start_vector,
        final_emission_matrix,
        final_transition_matrix
    )

def assert_is_row_stochastic_matrix(matrix: numpy.ndarray):
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
def ndarray_list():
    data = [
        numpy.random.randint(low=0, high=10, size=5, dtype=int),
        numpy.random.randint(low=5, high=100, size=500, dtype=int),
        numpy.random.randint(low=567, high=869, size=88, dtype=int)
    ]
    return data
    

@pytest.fixture(params=[(5, 16), (4, 128), (5, 128), (4, 128), (4, 128), (3, 19), (4, 128)])
def random_hmm_params(request):
    n_states, n_symbols = request.param
    start_vector, emission_matrix, transition_matrix = random_left_right_hmm_params(n_states, n_symbols)

    hmm_params = HmmParams(start_vector, emission_matrix, transition_matrix)
    return hmm_params

@pytest.fixture
def observation_sequences(random_hmm_params: HmmParams):
    n_states, n_symbols = random_hmm_params.emission_matrix.shape

    observation_sequence_sizes = [3, 7, 13, 43, 9, 8, 5, 5, 231, 48, 97, 89, 100, 27]
    observation_sequences = []
    for size in observation_sequence_sizes:
        obs = numpy.random.randint(low=0, high=n_symbols, size=size, dtype=int)
        observation_sequences.append(obs)
    
    observation_sequences = multiple_observation_sequences_from_ndarray_list(observation_sequences)

    return observation_sequences

    





# def test_multiple_observation_sequences_from_ndarray_list(ndarray_list):
#     mult_obs = bw.multiple_observation_sequences_from_ndarray_list(ndarray_list)

#     slices, arrays, length = mult_obs
#     print(slices)
#     print(arrays)

#     for i in range(len(ndarray_list)):
#         assert slices[i+1] - slices[i] == len(ndarray_list[i])
    
#     expected_total_length = sum([len(x) for x in ndarray_list])
#     assert expected_total_length == slices[-1]
#     assert len(ndarray_list) == length

#     for i in range(len(ndarray_list)):
#         start = slices[i]
#         stop = slices[i+1]
#         assert numpy.array_equal(arrays[start:stop], ndarray_list[i]) 


    

    # create a observation sequence

def test_train(random_hmm_params, observation_sequences):
    pi, b, a = random_hmm_params
    pi_r, b_r, a_r, log_prob = bw.train(pi, b, a, observation_sequences, 10)


    assert pi.shape == pi_r.shape
    assert b.shape == b_r.shape
    assert a.shape == a_r.shape

    assert_is_row_stochastic_matrix(pi_r)
    assert_is_row_stochastic_matrix(b_r)
    assert_is_row_stochastic_matrix(a_r)

    assert_is_log_prob(log_prob)


def test_calc_log_prob(random_hmm_params, observation_sequences):
    slices, array, length = observation_sequences
    start = slices[0]
    stop = slices[1]
    first_observation_seq = array[start: stop]

    pi, b, a = random_hmm_params

    log_prob = bw.calc_log_prob(pi, b, a, first_observation_seq)
    assert_is_log_prob(log_prob)

def test_calc_total_log_prob(random_hmm_params, observation_sequences):
    total_log_prob = bw.calc_total_log_prob(random_hmm_params, observation_sequences)
    assert_is_log_prob(total_log_prob)

def test_calc_mean_log_prob(random_hmm_params, observation_sequences):
    mean_log_prob = bw.calc_mean_log_prob(random_hmm_params, observation_sequences)
    assert_is_log_prob(mean_log_prob)


class TestBaumWelch:
    number_of_iterations = 100
    initial_log_probability = -165097.29
    final_log_probability = -137305.28

    # def test_calc_alpha_scaled(
    #     self,
    #     stamp_initial_hmm_params: HmmParams,
    #     stamp_observation_sequence: numpy.ndarray):

    #     start_vector, emission_matrix, transition_matrix = stamp_initial_hmm_params

    #     alpha, scalars = bw.calc_alpha_scaled(
    #         stamp_observation_sequence,
    #         start_vector,
    #         emission_matrix,
    #         transition_matrix)

    #     assert scalars.shape == (len(stamp_observation_sequence),)
    #     assert alpha.shape == (len(stamp_observation_sequence), transition_matrix.shape[0])
    #     assert_is_row_stochastic_matrix(alpha)

    #     self.alpha = alpha
    #     self.scalars = scalars

    # def test_calc_beta_scaled(
    #     self,
    #     stamp_initial_hmm_params: HmmParams,
    #     stamp_observation_sequence: numpy.ndarray):

    #     start_vector, emission_matrix, transition_matrix = stamp_initial_hmm_params

    #     beta = bw.calc_beta_scaled(
    #         stamp_observation_sequence,
    #         emission_matrix,
    #         transition_matrix,
    #         self.scalars
    #     )

    #     print(beta)

    #     assert beta.shape == (len(stamp_observation_sequence), transition_matrix.shape[0])
    #     assert_is_row_stochastic_matrix(beta)

    # def test_calc_gamma(
    #     self,
    #     stamp_initial_hmm_params: HmmParams,
    #     stamp_observation_sequence: numpy.ndarray):

    #     pi, b, a = stamp_initial_hmm_params

    #     alpha, scalars = bw.calc_alpha_scaled(stamp_observation_sequence, pi, b, a)
    #     beta = bw.calc_beta_scaled(stamp_observation_sequence, b, a, scalars)

    #     gamma = bw.calc_gamma(alpha, beta)

    #     assert gamma.shape == (len(stamp_observation_sequence), a.shape[0])

    #     # Rabiner (28)
    #     # For every t the sum over all i has to equal 1
    #     expected_sum = numpy.ones(len(stamp_observation_sequence))
    #     assert numpy.allclose(numpy.sum(gamma, axis=1), expected_sum)

    

    # @pytest.mark.skip(reason="Takes too long")
    def test_stamp(
        self,
        stamp_initial_hmm_params: HmmParams,
        stamp_observation_sequence: numpy.ndarray):
        
        pi, b, a = stamp_initial_hmm_params

        # for i in range(100):
        #     pi, b, a, log_prob = bw.reestimate(
        #     pi,
        #     b,
        #     a,
        #     stamp_observation_sequence
        # )
        # pi, b, a, log_prob = bw.reestimate(
        # pi,
        # b,
        # a,
        # stamp_observation_sequence)
        all_observations = bw.multiple_observation_sequences_from_ndarray_list([stamp_observation_sequence])
        assert all_observations.arrays.dtype == int
        pi, b, a, log_prob = bw.train(pi, b, a, all_observations, n_iterations=100)

        print(pi)
        print(b)
        print(a)
        assert_is_row_stochastic_matrix(a)
        assert_is_row_stochastic_matrix(b)
        # assert_is_row_stochastic_matrix(pi)
        assert log_prob == -165139.87961849626
        assert False

    # def test_final_log_probability(
    #     self,
    #     stamp_initial_hmm_params: HmmParams,
    #     stamp_final_hmm_params: HmmParams,
    #     stamp_observation_sequence: numpy.ndarray,
    #     ):

    #     my_final_hmm_params = bw.baum_welch(
    #         stamp_initial_hmm_params, 
    #         [stamp_observation_sequence], 
    #         self.number_of_iterations)

    #     my_final_log_probability = bw.calc_log_prob(my_final_hmm_params, stamp_observation_sequence)

    #     assert my_final_log_probability == self.final_log_probability
    


# @pytest.fixture
# def alpha(stamp_initial_hmm_params: HmmParams, stamp_observation_sequence: numpy.ndarray):
#     start_vector, emission_matrix, transition_matrix = stamp_initial_hmm_params
#     alpha, _ = bw.calc_alpha_scaled(
#         stamp_observation_sequence,
#         start_vector,
#         emission_matrix,
#         transition_matrix)

#     return alpha

# @pytest.fixture
# def beta(stamp_initial_hmm_params: HmmParams, stamp_observation_sequence: numpy.ndarray):
#     start_vector, emission_matrix, transition_matrix = stamp_initial_hmm_params
#     beta = bw.calc_beta_scaled(
#         stamp_observation_sequence,
#         emission_matrix,
#         transition_matrix
#     )

#     return beta






# def test_calc_alpha_scaled(
#     stamp_initial_hmm_params: HmmParams, 
#     stamp_observation_sequence: numpy.ndarray):
    
#     start_vector, emission_matrix, transition_matrix = stamp_initial_hmm_params

#     alpha, log_probability = bw.calc_alpha_scaled(
#         stamp_observation_sequence,
#         start_vector,
#         emission_matrix,
#         transition_matrix
#         )

#     assert_is_row_stochastic_matrix(alpha)
#     assert alpha.shape == (len(stamp_observation_sequence), transition_matrix.shape[0])
#     assert not math.isnan(log_probability)
#     assert log_probability < 1

# def test_calc_beta_scaled(
#     stamp_initial_hmm_params: HmmParams,
#     stamp_observation_sequence: numpy.ndarray):

#     start_vector, emission_matrix, transition_matrix = stamp_initial_hmm_params

#     beta = bw.calc_beta_scaled(stamp_observation_sequence, emission_matrix, transition_matrix)
#     print(beta)

#     assert beta.shape == (len(stamp_observation_sequence), transition_matrix.shape[0])
#     assert_is_row_stochastic_matrix(beta[:len(stamp_observation_sequence) - 1, :])

# def test_calc_gamma(gamma, stamp_initial_hmm_params: HmmParams, stamp_observation_sequence):
#     transition_matrix = stamp_initial_hmm_params.transition_matrix
#     assert gamma.shape == (len(stamp_observation_sequence), transition_matrix.shape[0])
#     assert_is_row_stochastic_matrix(gamma)






