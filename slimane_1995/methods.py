from hmm.params import conditional_random_ergodic_hmm_params
import hmm.bw as bw
from ga.numba_ga import GaHMM
import numpy
npr = numpy.random.default_rng()
import ga.selection as selection
from data.data import MultipleObservationSequences

N_BW_ITERATIONS = 50
N_GA_ITERATIONS = 200

def uniform_mutation_factory(mutation_chance: float):

    def mutation_func(chromosome, gabw):
        for i in range(len(chromosome) - 4):
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

def do_bw_only(n_states: int, n_symbols, observation: MultipleObservationSequences) -> float:
    hmm_params = conditional_random_ergodic_hmm_params(n_states, n_symbols)
    reestimated_hmm_params , log_prob_trace = bw.train_single_hmm(hmm_params, observation, n_iterations=N_BW_ITERATIONS)
    final_log_prob = log_prob_trace[-1]
    return final_log_prob
    
def do_ga_and_bw(n_states: int, n_symbols: int, observation: MultipleObservationSequences) -> float:
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
    

    _, best_fitness = gabw.start()
    return best_fitness

def do_ga_and_bw_return_gabw(n_states: int, n_symbols: int, observation):
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
    _, best_fitness = gabw.start()
    return gabw

def do_ga_only(n_states: int, n_symbols: int, observation: MultipleObservationSequences) -> float:
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
    _, best_fitness = gabw.start()
    return best_fitness

def do_random_only(n_states: int, n_symbols: int, observation: MultipleObservationSequences) -> float:
    hmm_params = conditional_random_ergodic_hmm_params(n_states, n_symbols)
    log_prob = bw.calc_total_log_prob(hmm_params, observation)
    return log_prob