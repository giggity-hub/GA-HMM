from ga.numba_ga import GaHMM
from hmm.hmm import random_left_right_hmm_params2
from ga.fitness import numba_mean_log_prob_fitness
from data.digits import load_dataset
from data.digits import load_dataset
from test.conftest import assert_is_row_stochastic, assert_is_log_prob
import numpy



def assert_population_is_row_stochastic(gabw_obj: GaHMM):
    for i in range(gabw_obj.population_size):
        
        hmm_params = gabw_obj.chromosome2hmm_params(gabw_obj.population[i])
        print(hmm_params.transition_matrix)
        assert_is_row_stochastic(hmm_params.start_vector)
        assert_is_row_stochastic(hmm_params.emission_matrix)
        assert_is_row_stochastic(hmm_params.transition_matrix)


def test_init(gabw: GaHMM):
    assert_population_is_row_stochastic(gabw)

def test_initialize_chromosome_mask(gabw: GaHMM):
    assert gabw.chromosome_mask.shape == (gabw.population.shape[1],)
    

def test_calculate_slices(gabw: GaHMM):
    n_states = gabw.n_states
    n_symbols = gabw.n_symbols

    slices = gabw.slices

    assert slices.start_probs.start == 0
    assert slices.start_probs.stop == n_states
    assert slices.start_probs.stop == slices.emission_probs.start
    assert slices.emission_probs.stop == slices.transition_probs.start
    assert slices.transition_probs.stop == slices.fitness.start
    assert slices.fitness.stop == slices.rank.start

    assert slices.start_probs.step == n_states
    assert slices.emission_probs.step == n_symbols
    assert slices.transition_probs.step == n_states

    assert slices.rank.stop == n_states + n_states*n_symbols + n_states**2 + 2

    assert slices.start_probs.stop - slices.start_probs.start == n_states
    assert slices.emission_probs.stop - slices.emission_probs.start == n_states*n_symbols
    assert slices.transition_probs.stop - slices.transition_probs.start == n_states*n_states



def test_hmm_params2chromosome(gabw: GaHMM):
    n_states = gabw.n_states
    n_symbols = gabw.n_symbols
    hmm_params = random_left_right_hmm_params2(n_states, n_symbols)
    chromosome = gabw.hmm_params2chromosome(hmm_params)

    start_vector = chromosome[0: n_states]
    emission_vector = chromosome[n_states : (n_states + n_states * n_symbols)]
    transition_vector = chromosome[(n_states + n_states * n_symbols) : ((n_states + n_states * n_symbols) + n_states**2)]

    assert numpy.array_equal(start_vector, hmm_params.start_vector)
    assert numpy.array_equal(emission_vector, hmm_params.emission_matrix.flatten())
    assert numpy.array_equal(transition_vector, hmm_params.transition_matrix.flatten())

def test_chromosome2hmm_params(gabw: GaHMM):
    pop_size, n_genes = gabw.population.shape
    for i in range(pop_size):
        hmm_params = gabw.chromosome2hmm_params(gabw.population[i])

        assert hmm_params.start_vector.shape == (gabw.n_states,)
        assert hmm_params.emission_matrix.shape == (gabw.n_states, gabw.n_symbols)
        assert hmm_params.transition_matrix.shape == (gabw.n_states, gabw.n_states)

        assert_is_row_stochastic(hmm_params.start_vector)
        assert_is_row_stochastic(hmm_params.emission_matrix)
        assert_is_row_stochastic(hmm_params.transition_matrix)

        start, stop, _ = gabw.slices.start_probs
        assert numpy.array_equal(hmm_params.start_vector, gabw.population[i, start: stop])
        start, stop, _ = gabw.slices.emission_probs
        assert numpy.array_equal(hmm_params.emission_matrix.flatten(), gabw.population[i, start:stop])
        start, stop, _ = gabw.slices.transition_probs
        assert numpy.array_equal(hmm_params.transition_matrix.flatten(), gabw.population[i, start: stop])

def test_chromosome2hmm_params_and_hmm_params2chromosome(gabw: GaHMM):
    pop_size, n_genes = gabw.population.shape

    for i in range(pop_size):
        hmm_params = gabw.chromosome2hmm_params(gabw.population[i])
        chromosome = gabw.hmm_params2chromosome(hmm_params)
        assert chromosome.shape == gabw.population[i].shape
        # Omit last 2 genes from equality check because rank and fitness get lest when translating to hmm_params Tuple
        assert numpy.array_equal(chromosome[:-2], gabw.population[i][:-2])


def test_calc_fitness(gabw: GaHMM):
    gabw.calculate_fitness()

    for i in range(gabw.population_size):
        log_prob =  gabw.population[i][gabw.slices.fitness.start]
        assert_is_log_prob(log_prob)
    
    assert_population_is_row_stochastic(gabw)
    
def test_sort_population(gabw: GaHMM):
    random_fitness_values = numpy.random.uniform(-300, -200, size=gabw.population_size)
    sorted_fitness_values = numpy.flip(numpy.sort(random_fitness_values))

    gabw.population[:, gabw.slices.fitness.start] = random_fitness_values
    print(gabw.population[:, gabw.slices.fitness.start])

    gabw.sort_population()

    

    assert numpy.array_equal(sorted_fitness_values, gabw.population[:, gabw.slices.fitness.start])
    
    # assert False
    assert_population_is_row_stochastic(gabw)




def test_normalize_chromosomes(gabw: GaHMM):
    population_size, n_genes = gabw.population.shape
    gabw.population = numpy.random.rand(population_size, n_genes)

    gabw.normalize_chromosomes()

    # Assert Keep Shape
    assert gabw.population.shape == (population_size, n_genes)


    assert_population_is_row_stochastic(gabw)
        

def test_start(gabw: GaHMM, numba_disable_jit, numba_developer_mode):
    gabw.start()

    # n_states = 5
    # n_symbols = 16
    # population_size = 20
    # param_generator_func = random_left_right_hmm_params
    # slices = GaHMM.calculate_slices(n_states, n_symbols)
    # population = GaHMM.initialize_population(
    #     slices,
    #     n_states,
    #     n_symbols,
    #     population_size,
    #     param_generator_func
    # )

# def test_init(gabw):
    



# def test_calc_fitness():
#     n_states = 5
#     n_symbols = 16
#     population_size = 20
#     param_generator_func = random_left_right_hmm_params
#     slices = GaHMM.calculate_slices(n_states, n_symbols)
#     population = GaHMM.initialize_population(
#         slices,
#         n_states,
#         n_symbols,
#         population_size,
#         param_generator_func
#     )

#     training_data = load_dataset(dataset='train')
#     digit = 0
#     n_samples = 12
#     samples = training_data[digit][:n_samples]
#     fitness_func = numba_mean_log_prob_fitness(samples)

#     GaHMM.calculate_fitness(population, slices, fitness_func)
