import pytest
from typing import List, Tuple
import numpy
import pandas as pd
from hmm.types import HmmParams
import hmm.bw as bw
from test.assertions import assert_hmm_params_are_equal
import seaborn as sns
import matplotlib.pyplot as plt
from pytest import approx
import os
from hmm.hmm import multiple_observation_sequences_from_ndarray_list

@pytest.fixture
def stamp_observation_sequence():
    observation_symbols = []
    

    with open('test/baum_welch/Observation_sequence.txt') as f:
        line = f.readline()

        while line and len(line) >= 3:
            observation_symbol_index = line[2:]
            observation_symbols.append(int(observation_symbol_index))

            line = f.readline()

    return numpy.array(observation_symbols)


@pytest.fixture
def stamp_hmm_param_data_frames():
    filepath_Pi = 'test/baum_welch/stamp_Pi.csv'
    filepath_B = 'test/baum_welch/stamp_B.csv'
    filepath_A = 'test/baum_welch/stamp_A.csv'

    df_Pi = pd.read_csv(filepath_Pi, sep=' ')
    df_A = pd.read_csv(filepath_A ,sep=' ')
    df_B = pd.read_csv(filepath_B, sep=' ')

    return (df_Pi, df_B, df_A)


@pytest.fixture
def stamp_initial_hmm_params(stamp_hmm_param_data_frames):
    df_Pi, df_B, df_A = stamp_hmm_param_data_frames

    initial_start_vector = df_Pi.loc[:, 'initial_s1':'initial_s2'].to_numpy().flatten()
    initial_emission_matrix = df_B.loc[:, 'initial_s1':'initial_s2'].to_numpy().T
    initial_transition_matrix = df_A.loc[:, 'initial_s1':'initial_s2'].to_numpy().T

    return HmmParams(
        initial_start_vector,
        initial_emission_matrix,
        initial_transition_matrix
    )
    
@pytest.fixture
def stamp_final_hmm_params(stamp_hmm_param_data_frames):
    df_Pi, df_B, df_A = stamp_hmm_param_data_frames

    final_start_vector = df_Pi.loc[:, 'final_s1': 'final_s2'].to_numpy().flatten()
    final_emission_matrix = df_B.loc[:, 'final_s1': 'final_s2'].to_numpy().T
    final_transition_matrix = df_A.loc[:, 'final_s1':'final_s2'].to_numpy().T 

    return HmmParams(
        final_start_vector,
        final_emission_matrix,
        final_transition_matrix
    )

@pytest.fixture
def stamp_final_log_probability():
    return -137305.28


@pytest.mark.skip("Takes Too Long")
def test_train_single_observation(
    stamp_initial_hmm_params: HmmParams,
    stamp_final_hmm_params: HmmParams,
    stamp_final_log_probability: float,
    stamp_observation_sequence: numpy.ndarray):

    my_final_hmm_params, log_prob_trace = bw.train_single_observation(stamp_initial_hmm_params, stamp_observation_sequence, n_iterations=100)
    my_final_log_probability = log_prob_trace[-1]

    assert my_final_log_probability == approx(stamp_final_log_probability)
    assert_hmm_params_are_equal(my_final_hmm_params, stamp_final_hmm_params, atol=1e-04)

@pytest.mark.skip("Takes Too Long")
def test_train_multiple_observations(
    stamp_initial_hmm_params: HmmParams,
    stamp_final_hmm_params: HmmParams,
    stamp_final_log_probability: float,
    stamp_observation_sequence: numpy.ndarray):
    
    observation_sequence = multiple_observation_sequences_from_ndarray_list([stamp_observation_sequence])
    my_final_hmm_params, log_prob_trace = bw.train_multiple_observations(stamp_initial_hmm_params, observation_sequence, n_iterations=100)
    my_final_log_probability = log_prob_trace[-1]

    assert my_final_log_probability == approx(stamp_final_log_probability)
    assert_hmm_params_are_equal(my_final_hmm_params, stamp_final_hmm_params, atol=1e-04)




def calculate_difference_and_save_as_heat_map(
    ndarr_1: numpy.ndarray,
    ndarr_2: numpy.ndarray,
    output_path: str,
    figsize: Tuple[int, int] = (10, 10)
    ):

    difference = numpy.atleast_2d(ndarr_1 - ndarr_2)
    absolute_difference = numpy.absolute(difference)
    
    plt.subplots(figsize=figsize)
    sns.heatmap(absolute_difference.T, annot=True)
    plt.savefig(output_path)


def create_heatmap_of_hmm_param_difference(
    my_final_hmm_params_and_log_probability: Tuple[HmmParams, float], 
    stamp_final_hmm_params: HmmParams):

    my_final_hmm_params, _ = my_final_hmm_params_and_log_probability

    heat_map_output_folder = 'test/baum_welch/heatmaps'

    calculate_difference_and_save_as_heat_map(
        my_final_hmm_params.start_vector,
        stamp_final_hmm_params.start_vector,
        output_path = os.path.join(heat_map_output_folder, 'start_vector.png')
    )

    calculate_difference_and_save_as_heat_map(
        my_final_hmm_params.emission_matrix,
        stamp_final_hmm_params.emission_matrix,
        output_path = os.path.join(heat_map_output_folder, 'emission_matrix.png'),
        figsize=(7,20)
    )
    
    calculate_difference_and_save_as_heat_map(
        my_final_hmm_params.transition_matrix,
        stamp_final_hmm_params.transition_matrix,
        output_path = os.path.join(heat_map_output_folder, 'transition_matrix.png')
    )
