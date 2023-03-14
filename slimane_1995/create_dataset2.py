import itertools
from slimane_1995.observations import OBSERVATION_SEQUENCES, N_SYMBOLS_FOR_OBSERVATION_SEQUENCE
from hmm.params import conditional_random_ergodic_hmm_params
import numpy
from ga.numba_ga import GaHMM
npr = numpy.random.default_rng()
import hmm.bw as bw
import slimane_1995.methods as methods
import timeit
import pandas


N_STATES = numpy.arange(2, 21)
OBSERVATION_SEQUENCE_INDICES = numpy.arange(len(OBSERVATION_SEQUENCES))
N_ITERATIONS_EACH = 50
ITERATIONS = numpy.arange(N_ITERATIONS_EACH)

methods = {
    "bw_only": methods.do_bw_only,
    "ga_and_bw": methods.do_ga_and_bw,
    "random_only": methods.do_random_only,
    "ga_only": methods.do_ga_only
}

variables =  itertools.product(N_STATES, OBSERVATION_SEQUENCE_INDICES, ITERATIONS, methods.keys())

dataframe_rows = []
for (n_states, obs_index, i, method_name) in variables:
    fn = methods[method_name]
    observation_sequence = OBSERVATION_SEQUENCES[obs_index]
    n_symbols = N_SYMBOLS_FOR_OBSERVATION_SEQUENCE[obs_index]

    start_time = timeit.default_timer()
    log_prob = fn(n_states, n_symbols, observation_sequence)
    elapsed_time = timeit.default_timer() - start_time

    row = {
        'method': method_name,
        'observation_sequence': obs_index,
        'n_states': n_states,
        'time': elapsed_time,
        'log_prob': log_prob
    }
    dataframe_rows.append(row)

crossover_df = pandas.DataFrame(dataframe_rows)
crossover_df.to_csv('slimane_1995/dataframe2.csv')