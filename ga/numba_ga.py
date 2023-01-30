from lib.utils import rand_stochastic_vector, rand_stochastic_matrix, normalize_matrix, normalize_vector
import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Tuple, NamedTuple
from hmm.hmm import random_left_right_hmm_params2, ParamGeneratorFunction2
import lib.utils as utils
from numba import jit, njit
from hmm.types import HmmParams
from ga.types import (
    FitnessFunction, 
    CrossoverFunction, 
    MutationFunction, 
    SelectionFunction, 
    ChromosomeSlices, 
    ChromosomeMask, 
    SliceTuple,
    Chromosome)
import pytest

def assert_is_row_stochastic(matrix: numpy.ndarray):
    max_deviation = 1e-8
    matrix = numpy.atleast_2d(matrix) #For the case that a vector with only 1-Dimension is supplied
    assert numpy.sum(matrix, axis=1) == pytest.approx(numpy.ones(len(matrix)))
    assert numpy.min(matrix) >= 0
    assert numpy.max(matrix) < (1 + max_deviation)

def assert_chromosomes_are_row_stochastic(chromosomes: numpy.ndarray, gabw):
    for i in range(len(chromosomes)):
        hmm_params = gabw.chromosome2hmm_params(chromosomes[i])
        assert_is_row_stochastic(hmm_params.start_vector)
        assert_is_row_stochastic(hmm_params.emission_matrix)
        assert_is_row_stochastic(hmm_params.transition_matrix)

class Logs(NamedTuple):
    max: numpy.ndarray
    min: numpy.ndarray
    mean: numpy.ndarray
    total: numpy.ndarray


class GaHMM:
    param_generator_func: ParamGeneratorFunction2 = staticmethod(random_left_right_hmm_params2)
    fitness_func: FitnessFunction = None
    parent_select_func: SelectionFunction = None
    mutation_func: MutationFunction = None
    crossover_func: CrossoverFunction = None
    n_parents_per_mating: int = 2
    n_children_per_mating: int = 1
    keep_elitism: int = 1
    normalize_after_mutation: bool = True
    current_generation: int = 0

    def __init__(
        self,
        n_symbols: int,
        n_states: int,
        population_size: int,
        n_generations: int,
        ) -> None:

        # Parametrized Attributes
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.population_size = population_size
        self.n_generations = n_generations

        # Ab hier beginnt der Müll:
        self.n_genes = self.calc_n_genes(n_states, n_symbols)
        self.slices = self.calculate_slices(self.n_states, self.n_symbols)
        self.population = self.initialize_population()
        # Assign Ranks to have default values for selection function
        self.assign_ranks_to_population()
        self.logs = self.initialize_logs()
        self.chromosome_mask = self.initialize_chromosome_mask()

        self.calc_n_parents_and_children_per_generation()


    def calc_n_parents_and_children_per_generation(self):
        self.n_children_per_generation = self.population_size - self.keep_elitism
        self.n_matings_per_generation = self.n_children_per_generation // self.n_children_per_mating
        self.n_parents_per_generation = self.n_matings_per_generation * self.n_parents_per_mating

    def validate(self):

        children_per_generation_is_multiple_of_children_per_mating = self.n_children_per_generation % self.n_children_per_mating == 0
        if not children_per_generation_is_multiple_of_children_per_mating:
            raise ValueError("Number of Children per Generation must be divisible by Number of Childrens Per Mating")

        


    def get_transition_probs_slice_for_state(self, state_index: int) -> SliceTuple:
        start, _, step = self.slices.emission_probs
        emissions_start = start + step * state_index
        emissions_stop = start + step * (state_index + 1)

        return SliceTuple(emissions_start, emissions_stop, step=1)

    def get_emission_probs_slice_for_state(self, state_index: int) -> SliceTuple:
        start, _, step = self.slices.transition_probs
        transition_start = start + step * state_index
        transition_stop = start + step * (state_index + 1)
        return SliceTuple(transition_start, transition_stop, step=1)

    def calc_n_genes(self, n_states: int, n_symbols: int) -> int:
        len_start_probs = n_states
        len_emission_probs = n_states*n_symbols
        len_transition_probs = n_states*n_states
        len_silent_states = 2 #One gene for fitness and one for Rank
        total_len =  len_start_probs + len_emission_probs + len_transition_probs + len_silent_states
        return total_len

    def initialize_chromosome_mask(self) -> ChromosomeMask:
        # Genes that have a 1 or zero stay the same during multiplication
        # gene_prod = numpy.prod(self.population, axis=0)
        # gene_sum = numpy.sum(self.population, axis=0)

        n_genes = self.population.shape[1]
        mask = numpy.zeros(n_genes, dtype=bool)

        for i in range(n_genes - 2):
            if self.population[0, i] == 0:
                mask[i] = True
            elif self.population[0,i] == 1:
                mask[i] = True

        mask[self.slices.fitness.start] = True
        mask[self.slices.rank.start] = True

        return mask
    # @njit
    # @staticmethod
    def calculate_slices(self, n_states: int, n_symbols: int) -> ChromosomeSlices:
        len_start_probs = n_states
        len_transition_probs = n_states*n_states
        len_emission_probs = n_states * n_symbols

        slice_start_probs = SliceTuple(0, len_start_probs, n_states)
        slice_emission_probs = SliceTuple(slice_start_probs.stop, slice_start_probs.stop +  len_emission_probs, n_symbols)
        slice_transition_probs = SliceTuple(slice_emission_probs.stop, slice_emission_probs.stop + len_transition_probs, n_states)

        slice_fitness = SliceTuple(slice_transition_probs.stop, slice_transition_probs.stop + 1, 1)
        slice_rank = SliceTuple(slice_fitness.stop, slice_fitness.stop + 1, 1)


        chromosome_slices = ChromosomeSlices(
            slice_start_probs,
            slice_emission_probs,
            slice_transition_probs,
            slice_fitness,
            slice_rank
        )

        return chromosome_slices
        
    # @njit
    # @staticmethod
    def initialize_population(self):

        chromosome_length = self.slices.rank.stop

        population = numpy.zeros((self.population_size, chromosome_length))

        for i in range(self.population_size):
            hmm_params = self.param_generator_func(self.n_states, self.n_symbols)
            chromosome = self.hmm_params2chromosome(hmm_params)
            population[i] = chromosome

        return population

        # return population
    
    def initialize_logs(self):
        max_arr = numpy.zeros(self.n_generations)
        min_arr = numpy.zeros(self.n_generations)
        mean_arr = numpy.zeros(self.n_generations)
        total_arr = numpy.zeros(self.n_generations)
        return Logs(max_arr, min_arr, mean_arr, total_arr)


    
    # @staticmethod
    def calculate_fitness(self):
        population_size = self.population.shape[0]
        n_states = self.slices.transition_probs.step
        n_symbols = self.slices.emission_probs.step

        for i in range(population_size):
            
            hmm_params = self.chromosome2hmm_params(self.population[i])
            log_prob = self.fitness_func(hmm_params)
            assert not numpy.isnan(log_prob)
            self.population[i, self.slices.fitness.start] = log_prob

        # return population

    def chromosome2hmm_params(self, chromosome: numpy.ndarray):
        n_states = self.slices.transition_probs.step
        n_symbols = self.slices.emission_probs.step

        start, stop, _ = self.slices.start_probs
        start_vector = chromosome[start:stop]

        start, stop, _ = self.slices.emission_probs
        emission_matrix = chromosome[start:stop].reshape((n_states, n_symbols))

        start, stop, _ = self.slices.transition_probs
        transition_matrix = chromosome[start:stop].reshape((n_states, n_states))

        hmm_params = HmmParams(start_vector.copy(), emission_matrix.copy(), transition_matrix.copy())
        return hmm_params

    def hmm_params2chromosome(self, hmm_params: HmmParams):
        n_genes = self.slices.rank.stop
        chromosome=numpy.zeros(n_genes)

        start, stop, _ = self.slices.start_probs
        chromosome[start: stop] = hmm_params.start_vector
        start, stop, _ = self.slices.emission_probs
        chromosome[start: stop] = hmm_params.emission_matrix.flatten()
        start, stop, _ = self.slices.transition_probs
        chromosome[start: stop] = hmm_params.transition_matrix.flatten()

        return chromosome


    # @staticmethod
    # @njit
    def sort_population(self):
        fitness_index = self.slices.fitness.start
        self.population = self.population[self.population[:, fitness_index].argsort()]
        self.population = numpy.flip(self.population, axis=0)
        # fitness_col = self.population[:, self.slices.fitness.start]
        # self.population = numpy.flip(self.population[fitness_col.argsort()])

    

    def update_logs(self):
        
        total_probability = 0
        min_probability = float('inf')
        max_probability = float('-inf')
        prob_sum = 0

        fitness_values = self.population[:, self.slices.fitness.start]
        gen_max = fitness_values.max()
        gen_min = fitness_values.min()
        gen_total = fitness_values.sum()
        gen_mean = gen_total / self.population_size

        self.logs.max[self.current_generation] = gen_max
        self.logs.min[self.current_generation] = gen_min
        self.logs.total[self.current_generation] = gen_total
        self.logs.mean[self.current_generation] = gen_mean


        # # Update Fitness and Rank
        # self.population.sort(reverse=True)
        # for i in range(self.population_size):
        #     chromosome = self.population[i]

        #     # numpy.exp(chromosome.probability) / prob_sum

        #     # chromosome.fitness = chromosome.probability/total_probability
        #     chromosome.rank = i

    
    def assign_ranks_to_population(self):
        # population_size = population.shape[0]
        self.population[:, self.slices.rank.start] = numpy.arange(0, self.population_size)

    def normalize_chromosomes(self):
        
        # assume that elites are still normalized
        self.normalize(self.slices.start_probs)
        self.normalize(self.slices.emission_probs)
        self.normalize(self.slices.transition_probs)

    def normalize(self, slice_tuple):
        
        start, stop, step = slice_tuple
        
        for i in range(start, stop, step):
            lo = i
            hi = i+step

            probs = self.population[:, lo: hi]
            
            probs_sum = numpy.sum(probs, axis=1)
            
            probs_sum_t = numpy.atleast_2d(probs_sum).T
            
            moped = (probs / probs_sum_t)

            self.population[:, lo:hi] = moped
    
    def do_selection_step(self):
        
        parents = self.parent_select_func(self.population, self.n_parents_per_generation, self.slices, self)
        return parents.copy()

    def do_crossover_step(self, parents):

        children = numpy.zeros((self.n_children_per_generation ,self.n_genes))
        child_index = 0

        for parent_index in range(0, self.n_parents_per_generation, self.n_parents_per_mating):
            
            par = parents[parent_index : (parent_index+ self.n_parents_per_mating), :].copy()
            childs = self.crossover_func(par, self.n_children_per_mating, self.slices, self)

            children[child_index:(child_index + len(childs)), : ] = childs

            child_index +=len(childs)

        
        if not child_index == len(children): raise Exception( "The Crossover Function does not support the provided n_children_per_mating" )


        children = self.assign_default_values_to_hidden_genes(children)

        return children

    def assign_default_values_to_hidden_genes(self, population):
        population[:, self.slices.rank.start] = 0
        population[:, self.slices.fitness.start]= float('-inf')
        return population
        


    def do_mutation_step(self, children):
        n_children = children.shape[0]
        

        for i in range(n_children):
            mutated_child = self.mutation_func(children[i, :], self.slices, self.chromosome_mask, self)
            children[i] = mutated_child.copy()
        
        return children.copy()

    def smooth_emission_probabilities(self):
        # Emission values can't be zero otherwise Baum-Welch doesn't work
        smoothing_value = 1e-10
        start, stop, _ = self.slices.emission_probs
        self.population[:, start:stop] = self.population[:, start: stop] + smoothing_value

    def next_iteration(self):
        # print(f'starting iteration {iteration}')
        for iteration in range(self.n_generations):
            self.current_generation = iteration

            assert not numpy.any(numpy.isnan(self.population))
            assert_chromosomes_are_row_stochastic(self.population, self)
            self.calculate_fitness()
            assert not numpy.any(numpy.isnan(self.population))
            assert_chromosomes_are_row_stochastic(self.population, self)
            self.sort_population()
            assert not numpy.any(numpy.isnan(self.population))
            assert_chromosomes_are_row_stochastic(self.population, self)
            self.assign_ranks_to_population()
            assert not numpy.any(numpy.isnan(self.population))
            assert_chromosomes_are_row_stochastic(self.population, self)

            self.update_logs()
            assert not numpy.any(numpy.isnan(self.population))
            assert_chromosomes_are_row_stochastic(self.population, self)

            parents = self.do_selection_step()
            assert not numpy.any(numpy.isnan(parents))
            assert_chromosomes_are_row_stochastic(parents, self)
            children_after_cross = self.do_crossover_step(parents)
            # Hier war Noch alles Gucci
            assert not numpy.any(numpy.isnan(children_after_cross))
            # assert_chromosomes_are_row_stochastic(children, self)
            # children = self.population[self.keep_elitism:, :].copy()
            children = self.do_mutation_step(children_after_cross.copy())
            assert not numpy.any(numpy.isnan(children))

            # assert_chromosomes_are_row_stochastic(children, self)

            # The next population is elites of current population plus children
            self.population[self.keep_elitism:, :] = children
            
            # self.population = self.population.copy()
            assert not numpy.any(numpy.isnan(self.population))
            # assert_chromosomes_are_row_stochastic(self.population, self)
            self.smooth_emission_probabilities()
            self.normalize_chromosomes()
            assert not numpy.any(numpy.isnan(self.population))
            assert_chromosomes_are_row_stochastic(self.population, self)

    def start(self):
        self.calc_n_parents_and_children_per_generation()
        self.validate()
        self.next_iteration()


    def plot(self):
        x = range(self.n_generations)
        
        plt.plot(x, self.logs.max, label='max')
        plt.plot(x, self.logs.mean, label='mean')
        plt.plot(x, self.logs.min, label='min')

        plt.legend()
        plt.show()



