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
from ga.types import FitnessFunction, CrossoverFunction, MutationFunction, SelectionFunction, ChromosomeSlices, ChromosomeMask, SliceTuple


class Logs(NamedTuple):
    max: numpy.ndarray
    min: numpy.ndarray
    mean: numpy.ndarray
    total: numpy.ndarray


class GaHMM:
    param_generator_func: ParamGeneratorFunction2 = random_left_right_hmm_params2
    fitness_func: FitnessFunction = None
    parent_select_func: SelectionFunction = None
    mutation_func: MutationFunction = None
    crossover_func: CrossoverFunction = None
    def __init__(
        self,
        n_symbols: int,
        n_states: int,
        population_size: int,
        n_generations: int,
        # fitness_func: FitnessFunction,
        # parent_select_func: SelectionFunction,
        # mutation_func: MutationFunction, 
        # crossover_func: CrossoverFunction,
        keep_elitism=1,
        normalize_after_mutation=True,
        param_generator_func: ParamGeneratorFunction2 = random_left_right_hmm_params2,
        ) -> None:

        # Parametrized Attributes
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.population_size = population_size
        self.n_generations = n_generations
        self.n_genes = self.calc_n_genes(n_states, n_symbols)
        # self.fitness_func = staticmethod(fitness_func)
        # self.parent_select_func: SelectionFunction = staticmethod(parent_select_func)
        # self.mutation_func: MutationFunction = staticmethod(mutation_func)
        # self.crossover_func = staticmethod(crossover_func)
        self.keep_elitism=keep_elitism
        self.normalize_after_mutation=normalize_after_mutation
        self.param_generator_func = staticmethod(param_generator_func)

        # Calculated Attributes
        self.offspring_count = self.population_size - self.keep_elitism
        self.current_generation = 0

        # self.population = [self.new_chromosome() for i in range(self.population_size)]
        self.slices = self.calculate_slices(self.n_states, self.n_symbols)
        # self.population = self.initialize_population(self.slices, self.n_states, self.n_symbols, self.population_size, self.param_generator_func)

        self.population = self.initialize_population()
        # Assign Ranks to have default values for selection function
        self.assign_ranks_to_population()
        self.logs = self.initialize_logs()
        self.chromosome_mask = self.initialize_chromosome_mask()

    def calc_n_genes(self, n_states: int, n_symbols: int) -> int:
        len_start_probs = n_states
        len_emission_probs = n_states*n_symbols
        len_transition_probs = n_states*n_states
        len_silent_states = 2 #One gene for fitness and one for Rank
        total_len =  len_start_probs + len_emission_probs + len_transition_probs + len_silent_states
        return total_len

    def initialize_chromosome_mask(self) -> ChromosomeMask:
        gene_sum = numpy.sum(self.population, axis=0)

        n_genes = self.population.shape[1]
        mask = numpy.zeros(n_genes, dtype=bool)

        for i in range(n_genes):
            mask[i] = gene_sum[i] == 0 or gene_sum[i] == 1

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
        parents = self.parent_select_func(self.population, self.offspring_count, self.slices, self)
        return parents

    def do_crossover_step(self, parents):

        children = numpy.empty((self.offspring_count ,self.n_genes))
        child_index = 0

        n_parents = parents.shape[0]
        parents_per_child = 2
        for parent_index in range(0, n_parents, parents_per_child):
            
            par = parents[parent_index : parent_index+parents_per_child]
            child = self.crossover_func(par, self.slices, self)
            children[child_index] = child

        return children

    def do_mutation_step(self, children):
        n_children = children.shape[0]

        for i in range(n_children):
            mutated_child = self.mutation_func(children[i], self.slices, self.chromosome_mask, self)
            children[i] = mutated_child
        
        return children

    def start(self):
        for iteration in range(self.n_generations):
            print(f'starting iteration {iteration}')
            self.current_generation = iteration

            self.calculate_fitness()
            self.sort_population()
            self.assign_ranks_to_population()

            self.update_logs()

            parents = self.do_selection_step()
            children = self.do_crossover_step(parents)
            children = self.do_mutation_step(children)

            # The next population is elites of current population plus children
            self.population[self.keep_elitism:, :] = children

            self.normalize_chromosomes()

        return self.logs

    def plot(self):
        x = range(self.n_generations)
        
        plt.plot(x, self.logs.max, label='max')
        plt.plot(x, self.logs.mean, label='mean')
        plt.plot(x, self.logs.min, label='min')

        plt.legend()
        plt.show()



