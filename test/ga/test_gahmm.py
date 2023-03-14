import pytest
from ga.numba_ga import GaHMM
from ga.numba_ga import normalize_population, calculate_fitness_of_population, create_population, sort_population_by_fitness_values
from ga.gabw_logger import GABWLogger

# from ga.mutation import constant_uniform_mutation_factory, delete_random_emission_symbols
# from ga.crossover import uniform_crossover
# from ga.selection import rank_selection_factory

import ga.mutation as mutation
import ga.crossover as crossover
import ga.selection as selection
from data.data import Dataset
import numpy
from ga.types import ChromosomeSlices, Population

from hmm.params import create_multiple_uniform_random_left_right_hmm_params
from hmm.types import MultipleHmmParams

import ga.representation as representation

from test.assertions import (
    assert_all_values_are_probabilities,
    assert_all_values_are_log_probabilities,
    assert_valid_hmm_params,
    assert_valid_multiple_hmm_params,
    assert_hmm_params_are_within_tolerance,
    assert_multiple_hmm_params_are_equal
)





@pytest.fixture
def n_symbols():
    return 128

@pytest.fixture
def observation_sequences(n_symbols):
    dataset = Dataset('fsdd', n_symbols=128)
    obs = dataset.get_first_n_observations_of_category(0, 10)
    return obs

@pytest.fixture
def gabw(observation_sequences):
    gabw = GaHMM(
        n_symbols=128,
        n_states=4,
        population_size=13,
        n_generations=40,
        observations=observation_sequences
    )

    # gabw.fitness_func = mean_log_prob_fitness(observation_sequences)
    gabw.mutation_func = mutation.constant_uniform_mutation_factory(mutation_threshold=0.1)
    gabw.crossover_func = crossover.uniform_crossover
    gabw.parent_select_func = selection.rank_selection_factory(gabw.population_size)

    gabw.bake()
    return gabw


def test_gahmm_start(gabw: GaHMM):
    hmm_params, fitness = gabw.start()
    assert False

CROSSOVER_FUNCTIONS = [
    crossover.uniform_crossover,
    crossover.arithmetic_mean_crossover,
    crossover.n_point_crossover_factory(1),
    crossover.n_point_crossover_factory(2),
    crossover.n_point_crossover_factory(3),
]

@pytest.fixture(params=CROSSOVER_FUNCTIONS)
def crossover_func(request):
    return request.param

@pytest.fixture
def unnormalized_gabw(gabw: GaHMM):
    
    n_chromosomes, n_genes = gabw.population.shape
    unnormalized_population = numpy.random.rand(n_chromosomes, n_genes - 4)
    gabw.population[:, :(n_genes-4)] = unnormalized_population
    return gabw


@pytest.fixture
def parents(gabw: GaHMM):
    parents = gabw.do_selection_step()
    return parents



# def test_initialize_hmms(gabw: GaHMM):
#     assert gabw.hmms.Bs.shape == (gabw.population_size, gabw.n_states, gabw.n_symbols)
#     assert_valid_multiple_hmm_params(gabw.hmms)

# def test_do_mutation_step(gabw: GaHMM):
#     PIs_before = gabw.hmms.PIs.copy()
#     As_before = gabw.hmms.As.copy()
#     Bs_before = gabw.hmms.Bs.copy()

#     gabw.do_mutation_step(children=gabw.population)

#     assert numpy.array_equal(PIs_before, gabw.hmms.PIs)
#     assert numpy.array_equal(As_before, gabw.hmms.As)
#     assert not numpy.array_equal(Bs_before, gabw.hmms.Bs)


# def test_do_selection_step(gabw: GaHMM):
#     population_before = gabw.population.copy()

#     parents = gabw.do_selection_step()

#     assert parents.shape == (gabw.n_parents_per_generation, gabw.n_genes)

#     parents[:,:] = numpy.empty_like(parents)

#     assert numpy.array_equal(population_before, gabw.population)

# def test_do_crossover_step(gabw: GaHMM, parents: Population):
#     children = gabw.do_crossover_step(parents)
#     assert children.shape == (gabw.n_children_per_generation, gabw.n_genes)


# @pytest.mark.skip(reason="takes too long without numba")
# def test_start(gabw: GaHMM):
#     # gabw.n_generations = 0
#     gabw.n_bw_iterations_before_ga = 10

#     gabw.start()


def test_normalize_chromosomes(unnormalized_gabw: GaHMM):
    normalized = normalize_population(unnormalized_gabw.population)
    normalized_hmm_params = representation.population_as_multiple_hmm_params(normalized)

    assert_valid_multiple_hmm_params(normalized_hmm_params)


def test_calculate_fitness_of_population(gabw: GaHMM):
    fitness_values = calculate_fitness_of_population(gabw.population, gabw.observations)

    assert fitness_values.shape == (gabw.population_size, )
    assert_all_values_are_log_probabilities(fitness_values)


@pytest.fixture
def unsorted_population():
    population_size=30
    population = create_population(population_size, n_states=13, n_symbols=17)
    population[:, representation.FIELDS.FITNESS] = numpy.random.uniform(low=-300, high=-50, size=population_size)
    return population

def test_sort_population_by_fitness(unsorted_population):
    fitness_values = unsorted_population[:, representation.FIELDS.FITNESS]
    expected_fitness_values_after_sort = numpy.flip(numpy.sort(fitness_values))
    population_after_sort = sort_population_by_fitness_values(unsorted_population)
    actual_fitness_values_after_sort = population_after_sort[:, representation.FIELDS.FITNESS]

    assert numpy.array_equal(expected_fitness_values_after_sort, actual_fitness_values_after_sort)

# chromosome.fields[fields]


def test_do_crossover_step(gabw: GaHMM, crossover_func):
    gabw.crossover_func = crossover_func
    children = gabw.do_crossover_step(gabw.population)

    assert children.shape == (gabw.population_size // 2, gabw.n_genes)

    

    









# @pytest.mark.skip(reason="gerade nicht bruder")
# def test_train_with_bw_for_n_iterations(gabw: GaHMM):
#     gabw.n_generations = 0
#     gabw.n_bw_iterations_before_ga = 300
#     gabw.start()
#     assert_valid_multiple_hmm_params(gabw.hmms)


# @pytest.fixture
# def gabw(gabw_mock: GaHMM, observation_sequences):
#     gabw_mock.fitness_func = mean_log_prob_fitness(observation_sequences)
#     # gabw_mock.mutation_func = numba_constant_uniform_mutation2(mutation_threshold=0.1)
#     gabw_mock.mutation_func = delete_random_emission_symbols(n_zeros=1)
#     gabw_mock.crossover_func = uniform_crossover
#     gabw_mock.parent_select_func = rank_selection_factory(gabw_mock.population_size)
#     return gabw_mock



# depends on slices
# def test_initialize_population(gabw: GaHMM):
#     assert gabw.population.shape == (gabw.population_size, gabw.n_genes)
#     # The Fitness and Rank columns do not contain probability values
#     end_of_probabilities = gabw.slices.transition_probs.stop
#     assert_chromosomes_are_row_stochastic(gabw.population, gabw)
#     # assert_all_values_are_probabilities(gabw.population[:, :end_of_probabilities])


# @pytest.fixture
# def gabw_with_all_zero_ranks(gabw: GaHMM):
#     gabw.population[:, gabw.slices.rank.start] = numpy.zeros(gabw.population_size)
#     return gabw

# def test_assign_ranks_to_population(gabw_with_all_zero_ranks: GaHMM):
#     gabw_with_all_zero_ranks.assign_ranks_to_population()

#     expected_ranks = numpy.arange(gabw_with_all_zero_ranks.population_size)
#     rank_col = gabw_with_all_zero_ranks.slices.rank.start
#     actual_ranks = gabw_with_all_zero_ranks.population[:, rank_col]

#     assert numpy.array_equal(expected_ranks, actual_ranks)



# class TestChromosomeMask:
#     def test_chromosome_mask_shape(self, gabw: GaHMM):
#         assert gabw.chromosome_mask.shape == (gabw.n_genes, )

#     def test_chromosome_mask_contains_only_ones_and_zeros(self, gabw:GaHMM):
#         # All Masked entries that are not one must be zero
#         end_of_probabilities = gabw.slices.transition_probs.stop
#         only_masked_entries = (gabw.population * gabw.chromosome_mask)[:, :end_of_probabilities]
#         assert numpy.all((only_masked_entries==1)|(only_masked_entries==0))


# class TestCalculateSlices:
#     def test_slice_starts(self, gabw: GaHMM):
#         assert gabw.slices.start_probs.start == 0
#         assert gabw.slices.emission_probs.start == gabw.n_states
#         assert gabw.slices.transition_probs.start == gabw.n_states + gabw.n_states*gabw.n_symbols
    
#     def test_slice_steps(self, gabw: GaHMM):
#         assert gabw.slices.start_probs.step == gabw.n_states
#         assert gabw.slices.emission_probs.step == gabw.n_symbols
#         assert gabw.slices.transition_probs.step == gabw.n_states
    
#     def test_slice_stops(self, gabw: GaHMM):
#         assert gabw.slices.start_probs.stop == gabw.n_states
#         assert gabw.slices.emission_probs.stop == gabw.n_states + gabw.n_states*gabw.n_symbols 
#         assert gabw.slices.transition_probs.stop == gabw.n_states + gabw.n_states*gabw.n_symbols + gabw.n_states**2


# def test_calculate_fitness(gabw: GaHMM):
#     gabw.calculate_fitness()

#     fitness_values = gabw.population[:, gabw.slices.fitness.start]
#     assert_all_values_are_log_probabilities(fitness_values)


# @pytest.mark.skip(reason='Worked! But takes too long')
# def test_chromosome2hmm_params_and_hmm_params2chromosome(gabw: GaHMM):
#     pop_size, n_genes = gabw.population.shape

#     for i in range(pop_size):
#         hmm_params = gabw.chromosome2hmm_params(gabw.population[i])
#         chromosome = gabw.hmm_params2chromosome(hmm_params)
#         assert chromosome.shape == gabw.population[i].shape
#         # Omit last 2 genes from equality check because rank and fitness get lost when translating to hmm_params Tuple
#         assert numpy.array_equal(chromosome[:-2], gabw.population[i][:-2])

# def test_population2multiple_hmm_params(gabw: GaHMM):
#     all_hmm_params = gabw.population2multiple_hmm_params()

#     first_start_vector = all_hmm_params.start_vectors[0]
#     first_emission_matrix = all_hmm_params.emission_matrices[0]
#     first_transition_matrix = all_hmm_params.transition_matrices[0]

#     first_hmm_params = HmmParams(first_start_vector, first_emission_matrix, first_transition_matrix)
#     assert_valid_hmm_params(first_hmm_params)

# def test_multiple_hmm_params2population(gabw: GaHMM):
#     population_copy = gabw.population.copy()

#     all_hmm_params = gabw.population2multiple_hmm_params()
#     gabw.multiple_hmm_params2population(all_hmm_params)

#     assert numpy.array_equal(population_copy, gabw.population)

# def test_train_population_with_baum_welch(gabw: GaHMM, observation_sequences):
#     gabw.train_population_with_baum_welch(observation_sequences, n_iterations=2)



# def test_sort_population(gabw: GaHMM):
#     random_fitness_values = numpy.random.uniform(-300, -200, size=gabw.population_size)
#     sorted_fitness_values = numpy.flip(numpy.sort(random_fitness_values))

#     gabw.population[:, gabw.slices.fitness.start] = random_fitness_values

#     gabw.sort_population()

#     assert numpy.array_equal(sorted_fitness_values, gabw.population[:, gabw.slices.fitness.start])
    

# def test_normalize_chromosomes(gabw: GaHMM):
#     population_size, n_genes = gabw.population.shape
#     gabw.population = numpy.random.rand(population_size, n_genes)

#     gabw.normalize_chromosomes()

#     # Assert Keep Shape
#     assert gabw.population.shape == (population_size, n_genes)


#     assert_chromosomes_are_row_stochastic(gabw.population, gabw)

# def test_do_selection(gabw: GaHMM):
#     parents = gabw.do_selection_step()
#     assert parents.shape == (gabw.n_parents_per_generation, gabw.n_genes)

#     assert_chromosomes_are_row_stochastic(parents, gabw)
#     assert_chromosomes_are_row_stochastic(gabw.population, gabw)

# @pytest.fixture
# def parents(gabw: GaHMM):
#     return gabw.do_selection_step()

# @pytest.fixture
# def children(parents, gabw: GaHMM):
#     return gabw.do_crossover_step(parents)

# @pytest.fixture
# def mutated_children(children, gabw: GaHMM):
#     return gabw.do_mutation_step(children)

# def test_children_shape(children, gabw: GaHMM):
#     assert children.shape == (gabw.n_children_per_generation, gabw.n_genes)

# def test_child_genes_are_probabilities(children, gabw: GaHMM):
#     children_without_silent_genes = children[:, :-2]
#     assert_all_values_are_probabilities(children_without_silent_genes)

# def test_child_fitness_has_been_reset(children, gabw: GaHMM):
#     fitness_values = children[:, gabw.slices.fitness.start]
#     assert numpy.all(fitness_values == float('-inf'))

# def test_child_rank_has_been_reset(children, gabw: GaHMM):
#     rank_values = children[:, gabw.slices.rank.start]
#     assert numpy.all(rank_values == 0)


# def test_mutation_kept_shape(children, mutated_children):
#     assert children.shape == mutated_children.shape

# def test_mutated_genes_are_probabilities(mutated_children):
#     genes_without_silent_genes = mutated_children[:, :-2]
#     assert_all_values_are_probabilities(genes_without_silent_genes)


# def test_start(gabw: GaHMM):
#     gabw.start()
