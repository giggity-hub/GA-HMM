import pytest
from ga.numba_ga import GaHMM
from ga.fitness import numba_mean_log_prob_fitness
from ga.mutation import numba_constant_uniform_mutation2, delete_random_emission_symbols
from ga.crossover import numba_single_point_crossover2
from ga.selection import rank_selection
import numpy
from ga.types import ChromosomeSlices
from hmm.bw_numba import multiple_observation_sequences_from_ndarray_list

from test.assertions import (
    assert_all_values_are_probabilities,
    assert_all_values_are_log_probabilities,
    assert_chromosomes_are_row_stochastic
)




@pytest.fixture
def gabw_mock():
    return GaHMM(
        n_symbols=128,
        n_states=4,
        population_size=13,
        n_generations=100
    )

@pytest.fixture
def observation_sequences(gabw_mock):
    sequences = []
    n_sequences = 10
    min_seq_length = 15
    max_seq_length = 500

    for i in range(n_sequences):
        seq_length = numpy.random.randint(low=min_seq_length, high=max_seq_length)
        rand_sequence = numpy.random.randint(low=0, high=gabw_mock.n_symbols, size=seq_length)
        sequences.append(rand_sequence)

    res = multiple_observation_sequences_from_ndarray_list(sequences)
    return res
    
@pytest.fixture
def gabw(gabw_mock: GaHMM, observation_sequences):
    gabw_mock.fitness_func = numba_mean_log_prob_fitness(observation_sequences)
    # gabw_mock.mutation_func = numba_constant_uniform_mutation2(mutation_threshold=0.1)
    gabw_mock.mutation_func = delete_random_emission_symbols(n_zeros=1)
    gabw_mock.crossover_func = numba_single_point_crossover2
    gabw_mock.parent_select_func = rank_selection(gabw_mock.population_size)
    return gabw_mock



# depends on slices
def test_initialize_population(gabw: GaHMM):
    assert gabw.population.shape == (gabw.population_size, gabw.n_genes)
    # The Fitness and Rank columns do not contain probability values
    end_of_probabilities = gabw.slices.transition_probs.stop
    assert_chromosomes_are_row_stochastic(gabw.population, gabw)
    # assert_all_values_are_probabilities(gabw.population[:, :end_of_probabilities])


@pytest.fixture
def gabw_with_all_zero_ranks(gabw: GaHMM):
    gabw.population[:, gabw.slices.rank.start] = numpy.zeros(gabw.population_size)
    return gabw

def test_assign_ranks_to_population(gabw_with_all_zero_ranks: GaHMM):
    gabw_with_all_zero_ranks.assign_ranks_to_population()

    expected_ranks = numpy.arange(gabw_with_all_zero_ranks.population_size)
    rank_col = gabw_with_all_zero_ranks.slices.rank.start
    actual_ranks = gabw_with_all_zero_ranks.population[:, rank_col]

    assert numpy.array_equal(expected_ranks, actual_ranks)



class TestChromosomeMask:
    def test_chromosome_mask_shape(self, gabw: GaHMM):
        assert gabw.chromosome_mask.shape == (gabw.n_genes, )

    def test_chromosome_mask_contains_only_ones_and_zeros(self, gabw:GaHMM):
        # All Masked entries that are not one must be zero
        end_of_probabilities = gabw.slices.transition_probs.stop
        only_masked_entries = (gabw.population * gabw.chromosome_mask)[:, :end_of_probabilities]
        assert numpy.all((only_masked_entries==1)|(only_masked_entries==0))


class TestCalculateSlices:
    def test_slice_starts(self, gabw: GaHMM):
        assert gabw.slices.start_probs.start == 0
        assert gabw.slices.emission_probs.start == gabw.n_states
        assert gabw.slices.transition_probs.start == gabw.n_states + gabw.n_states*gabw.n_symbols
    
    def test_slice_steps(self, gabw: GaHMM):
        assert gabw.slices.start_probs.step == gabw.n_states
        assert gabw.slices.emission_probs.step == gabw.n_symbols
        assert gabw.slices.transition_probs.step == gabw.n_states
    
    def test_slice_stops(self, gabw: GaHMM):
        assert gabw.slices.start_probs.stop == gabw.n_states
        assert gabw.slices.emission_probs.stop == gabw.n_states + gabw.n_states*gabw.n_symbols 
        assert gabw.slices.transition_probs.stop == gabw.n_states + gabw.n_states*gabw.n_symbols + gabw.n_states**2


def test_calculate_fitness(gabw: GaHMM):
    gabw.calculate_fitness()

    fitness_values = gabw.population[:, gabw.slices.fitness.start]
    assert_all_values_are_log_probabilities(fitness_values)


@pytest.mark.skip(reason='Worked! But takes too long')
def test_chromosome2hmm_params_and_hmm_params2chromosome(gabw: GaHMM):
    pop_size, n_genes = gabw.population.shape

    for i in range(pop_size):
        hmm_params = gabw.chromosome2hmm_params(gabw.population[i])
        chromosome = gabw.hmm_params2chromosome(hmm_params)
        assert chromosome.shape == gabw.population[i].shape
        # Omit last 2 genes from equality check because rank and fitness get lost when translating to hmm_params Tuple
        assert numpy.array_equal(chromosome[:-2], gabw.population[i][:-2])


def test_sort_population(gabw: GaHMM):
    random_fitness_values = numpy.random.uniform(-300, -200, size=gabw.population_size)
    sorted_fitness_values = numpy.flip(numpy.sort(random_fitness_values))

    gabw.population[:, gabw.slices.fitness.start] = random_fitness_values

    gabw.sort_population()

    assert numpy.array_equal(sorted_fitness_values, gabw.population[:, gabw.slices.fitness.start])
    

def test_normalize_chromosomes(gabw: GaHMM):
    population_size, n_genes = gabw.population.shape
    gabw.population = numpy.random.rand(population_size, n_genes)

    gabw.normalize_chromosomes()

    # Assert Keep Shape
    assert gabw.population.shape == (population_size, n_genes)


    assert_chromosomes_are_row_stochastic(gabw.population, gabw)

def test_do_selection(gabw: GaHMM):
    parents = gabw.do_selection_step()
    assert parents.shape == (gabw.offspring_count*2, gabw.n_genes)

    assert_chromosomes_are_row_stochastic(parents, gabw)
    assert_chromosomes_are_row_stochastic(gabw.population, gabw)

@pytest.fixture
def parents(gabw: GaHMM):
    return gabw.do_selection_step()

@pytest.fixture
def children(parents, gabw: GaHMM):
    return gabw.do_crossover_step(parents)

@pytest.fixture
def mutated_children(children, gabw: GaHMM):
    return gabw.do_mutation_step(children)

def test_children_shape(children, gabw: GaHMM):
    assert children.shape == (gabw.offspring_count, gabw.n_genes)

def test_child_genes_are_probabilities(children, gabw: GaHMM):
    children_without_silent_genes = children[:, :-2]
    assert_all_values_are_probabilities(children_without_silent_genes)

def test_child_fitness_has_been_reset(children, gabw: GaHMM):
    fitness_values = children[:, gabw.slices.fitness.start]
    assert numpy.all(fitness_values == float('-inf'))

def test_child_rank_has_been_reset(children, gabw: GaHMM):
    rank_values = children[:, gabw.slices.rank.start]
    assert numpy.all(rank_values == 0)


def test_mutation_kept_shape(children, mutated_children):
    assert children.shape == mutated_children.shape

def test_mutated_genes_are_probabilities(mutated_children):
    genes_without_silent_genes = mutated_children[:, :-2]
    assert_all_values_are_probabilities(genes_without_silent_genes)


def test_start(gabw: GaHMM):
    gabw.start()

    assert False