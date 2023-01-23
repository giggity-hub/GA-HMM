from ga.numba_ga import GaHMM
from hmm.hmm import random_left_right_hmm_params
from ga.fitness import numba_mean_log_prob_fitness
from data.digits import load_dataset
from data.digits import load_dataset


def test_calculate_slices():
    n_states = 5
    n_symbols = 16
    slices = GaHMM.calculate_slices(n_states, n_symbols)

    assert slices.start_probs.stop == slices.emission_probs.start
    assert slices.emission_probs.stop == slices.transition_probs.start
    assert slices.transition_probs.stop == slices.fitness.start
    assert slices.fitness.stop == slices.rank.start

    assert slices.rank.stop == n_states + n_states*n_symbols + n_states**2 + 2

    assert slices.start_probs.stop - slices.start_probs.start == n_states
    assert slices.emission_probs.stop - slices.emission_probs.start == n_states*n_symbols
    assert slices.transition_probs.stop - slices.transition_probs.start == n_states*n_states


    

def test_initialize_population():
    n_states = 5
    n_symbols = 16
    population_size = 20
    param_generator_func = random_left_right_hmm_params
    slices = GaHMM.calculate_slices(n_states, n_symbols)
    population = GaHMM.initialize_population(
        slices,
        n_states,
        n_symbols,
        population_size,
        param_generator_func
    )


def test_calc_fitness():
    n_states = 5
    n_symbols = 16
    population_size = 20
    param_generator_func = random_left_right_hmm_params
    slices = GaHMM.calculate_slices(n_states, n_symbols)
    population = GaHMM.initialize_population(
        slices,
        n_states,
        n_symbols,
        population_size,
        param_generator_func
    )

    training_data = load_dataset(dataset='train')
    digit = 0
    n_samples = 12
    samples = training_data[digit][:n_samples]
    fitness_func = numba_mean_log_prob_fitness(samples)

    GaHMM.calculate_fitness(population, slices, fitness_func)
