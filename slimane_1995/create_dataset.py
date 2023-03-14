from hmm.params import conditional_random_ergodic_hmm_params
import seaborn as sns
import numpy
from ga.numba_ga import GaHMM
npr = numpy.random.default_rng()
from data.data import observations
from typing import List
import hmm.bw as bw

import ga.selection as selection
import xarray as xr
from data.data import MultipleObservationSequences

def uniform_mutation_factory(mutation_chance: float):

    def mutation_func(chromosome, gabw):
        for i in range(len(chromosome - 2)):
            if npr.random() < mutation_chance:
                chromosome[i] = npr.random()
        return chromosome
    
    return mutation_func

def alpha_X_crossover(parents, n_children, gabw):
    pass

def one_X_crossover(parents, n_children, gabw):
    cutpoint = npr.choice(gabw.row_stochastic_cutpoints)

    child = parents[0].copy()
    child[cutpoint:] = parents[0, cutpoint:].copy()
    return numpy.atleast_2d(child)

OBSERVATION_SEQUENCES_LISTS = [
    [1,2,3,1,4,2,4,4],
    [1,2,2,1,1,1,2,2,2,1],
    [1,2,3,2,1,2,5,4,1,2,4],
    [1,1,1,2,2,1,2,3],
    [1,1,1,2,2,2,3,3,3],
    [1,2,3,1,2,3,1,2,3],
    [1,1,1,2,2,2,3,3,3,1,2,3],
    [1,1,2,2,3,3,4,4,1,2,3,4],
    [1,1,1,1,2,2,2,2],
    [1,2,3,4,5,6,6,5,4,3,2,1]
]

N_SYMBOLS_FOR_OBSERVATION_SEQUENCE = [max(obs) for obs in OBSERVATION_SEQUENCES_LISTS]

def load_observation_sequences():
    observation_sequences = []
    for i in range(len(OBSERVATION_SEQUENCES_LISTS)):
        obs_list = OBSERVATION_SEQUENCES_LISTS[i]
        obs_arr = numpy.array(obs_list)
        obs_arr_zero_based = obs_arr - 1
        obs_tuple = observations([obs_arr_zero_based])
        observation_sequences.append(obs_tuple)
    return observation_sequences 


observation_sequences = load_observation_sequences()

observation = observation_sequences[0]
n_symbols = observation.arrays.max() + 1
n_states = 10

N_BW_ITERATIONS = 50
N_GA_ITERATIONS = 200
N_OBSERVATION_SEQUENCES = 10
MIN_N_STATES = 2
MAX_N_STATES = 20
TOTAL_N_STATES = MAX_N_STATES - MIN_N_STATES + 1
N_STATES = numpy.arange(MIN_N_STATES, MAX_N_STATES + 1)

N_ITERATIONS = 100

print(len(N_STATES))


def do_bw_only(n_states: int, n_symbols, observation: MultipleObservationSequences) -> numpy.ndarray:
    hmm_params = conditional_random_ergodic_hmm_params(n_states, n_symbols)
    reestimated_hmm_params , log_prob_trace = bw.train_single_hmm(hmm_params, observation, n_iterations=N_BW_ITERATIONS)
    return log_prob_trace
    
def do_ga_and_bw(n_states: int, n_symbols: int, observation: MultipleObservationSequences) -> numpy.ndarray:
    gabw = GaHMM(
        n_symbols,
        n_states,
        population_size=60,
        n_generations=200,
        observations=observation,
        param_generator_func=conditional_random_ergodic_hmm_params
    )
    gabw.keep_elitism = 30
    gabw.parent_pool_size = 30
    # gabw.param_generator_func = conditional_random_ergodic_hmm_params
    gabw.mutation_func = uniform_mutation_factory(0.01)
    gabw.crossover_func = one_X_crossover
    gabw.parent_select_func = selection.random_selection
    gabw.n_bw_iterations_after_ga = N_BW_ITERATIONS
    gabw.start()
    return gabw.logs.logs.max(axis=0)

def do_random_only(n_states: int, n_symbols: int, observation: MultipleObservationSequences) -> numpy.ndarray:
    hmm_params = conditional_random_ergodic_hmm_params(n_states, n_symbols)
    log_prob = bw.calc_total_log_prob(hmm_params, observation)
    return log_prob


log_probs = {
    "bw_only" : numpy.zeros((N_OBSERVATION_SEQUENCES, TOTAL_N_STATES, N_ITERATIONS, N_BW_ITERATIONS)),
    # "bw_only_best_of_30" : numpy.zeros((N_OBSERVATION_SEQUENCES, TOTAL_N_STATES, N_ITERATIONS, N_BW_ITERATIONS)),
    "ga_only" : numpy.zeros((N_OBSERVATION_SEQUENCES, TOTAL_N_STATES, N_ITERATIONS, N_GA_ITERATIONS + 1)),
    "ga_and_bw" : numpy.zeros((N_OBSERVATION_SEQUENCES, TOTAL_N_STATES, N_ITERATIONS ,N_GA_ITERATIONS + N_BW_ITERATIONS + 1)),
    "random_only" : numpy.zeros((N_OBSERVATION_SEQUENCES, TOTAL_N_STATES, N_ITERATIONS, 1))
}

log_prob_methods = {
    "bw_only": do_bw_only,
    "ga_and_bw": do_ga_and_bw,
    "random_only": do_random_only
}
for method in ["bw_only", "ga_and_bw", "random_only"]:

    for obs_index in range(N_OBSERVATION_SEQUENCES):
        observation_seq = observation_sequences[obs_index]
        n_symbols = N_SYMBOLS_FOR_OBSERVATION_SEQUENCE[obs_index]
        for n_states_index in range(TOTAL_N_STATES):
            for iteration in range(N_ITERATIONS):
                n_states = N_STATES[n_states_index]

                fn = log_prob_methods[method]
                log_probs[method][obs_index, n_states_index, iteration] = fn(n_states, n_symbols, observation_seq)

                

DIMS = ['observation_sequence', 'n_states', 'iteration', 'time']
def save_nparray_as_xarray_netcdf(nparray, filename: str):
    xarr = xr.DataArray(nparray, dims=DIMS)
    xarr.to_netcdf(f"slimane_1995/{filename}.nc", mode='w')


log_probs["ga_only"] = log_probs["ga_and_bw"][:, :, :, :-N_BW_ITERATIONS]
for key, value in log_probs.items():
    save_nparray_as_xarray_netcdf(value, filename=key)