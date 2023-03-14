import numpy
from typing import Callable, List, Dict, Tuple, NamedTuple
from hmm.params import uniform_random_left_right_hmm_params, ParamGeneratorFunction2, MultipleObservationSequences
import lib.utils as utils
from numba import jit, njit
from numba.experimental import jitclass
import hmm.bw as bw
from ga.gabw_logger import GABWLogger
from lib.performance_timer import PerformanceTimer

import ga.representation as representation


from hmm.types import HmmParams, MultipleHmmParams
from ga.types import (
    FitnessFunction, 
    CrossoverFunction, 
    MutationFunction, 
    SelectionFunction, 
    ChromosomeSlices, 
    ChromosomeRanges,
    ChromosomeMask, 
    RangeTuple,
    Population,
    Chromosome)
import pytest


class Logs(NamedTuple):
    max: numpy.ndarray
    min: numpy.ndarray
    mean: numpy.ndarray
    total: numpy.ndarray

@jit
def normalize_population(population: Population):

    n_states = population[0, representation.FIELDS.N_STATES]
    n_symbols = population[0, representation.FIELDS.N_SYMBOLS]


    ranges = representation.calc_chromosome_ranges(n_states, n_symbols)
    for start, stop, step in ranges:
        for i in range(start, stop, step):
            population[:, i:(i+step)] = utils.normalize_array(population[:, i:(i+step)])

    return population

@jit
def calculate_fitness_of_population(population: Population, observations: MultipleObservationSequences):

    hmm_params = representation.population_as_multiple_hmm_params(population)
    fitness_values = bw.calc_total_log_prob_for_multiple_hmms(hmm_params, observations)
    return fitness_values


@jit
def train_population_with_bw(
    population: Population, 
    observations: MultipleObservationSequences, 
    n_iterations: int=1
    ) -> Population:
        
    hmm_params = representation.population_as_multiple_hmm_params(population)
    
    reestimated_hmm_params, log_prob_traces = bw.train_multiple_hmms(hmm_params, observations, n_iterations)

    new_population = representation.multiple_hmm_params_as_population(reestimated_hmm_params)

    fitness_values = log_prob_traces[:, -1]
    new_population[:, representation.FIELDS.FITNESS] = fitness_values
    return new_population


@jit
def sort_population_by_fitness_values(population: Population):

    fitness_values = population[:, representation.FIELDS.FITNESS]
    sorted_indices = numpy.flip(fitness_values.argsort())

    sorted_population = population[sorted_indices]
    sorted_population[:, representation.FIELDS.RANK] = numpy.arange(len(population))

    return sorted_population


def create_population(
        population_size: int, 
        n_states: int, 
        n_symbols: int, 
        param_generator_func: ParamGeneratorFunction2 = uniform_random_left_right_hmm_params):
    
    hmm_params_list = [param_generator_func(n_states, n_symbols) for i in range(population_size)]
    multiple_hmm_params = representation.hmm_params_list_as_multiple_hmm_params(hmm_params_list)
    population = representation.multiple_hmm_params_as_population(multiple_hmm_params)
    return population


# def apply_crossover_operator(parents: Population, crossover_func, n_parents_per_mating=2, n_children_per_mating=1):

#     n_parents, chromosome_length = parents.shape
#     n_children = (n_parents // n_parents_per_mating) * n_children_per_mating

#     children = numpy.zeros((n_children ,chromosome_length))

#     children_index = 0
#     parents_index = 0
#     for i in range(n_children):
#         parents_for_child = parents[parent_index:(parent_index + n_parents_per_mating)]
#         children_of_parents = crossover_func(parents, n_children_per_mating)

#         children_index += n_children_per_mating
#         parents_index += n_parents_per_mating



#     for parent_index in range(0, , self.n_parents_per_mating):
        
#         par = parents[parent_index : (parent_index+ self.n_parents_per_mating), :].copy()
#         childs = self.crossover_func(par, self.n_children_per_mating, self)

#         children[child_index:(child_index + self.n_children_per_mating), : ] = childs

#         child_index += self.n_children_per_mating

    
#     if not child_index == len(children): raise Exception( "The Crossover Function does not support the provided n_children_per_mating" )

#     return children








class GaHMM:
    # param_generator_func: ParamGeneratorFunction2 = staticmethod(uniform_random_left_right_hmm_params)
    parent_select_func: SelectionFunction = None
    mutation_func: MutationFunction = None
    crossover_func: CrossoverFunction = None

    n_parents_per_mating: int = 2
    n_children_per_mating: int = 1
    keep_elitism: int = 1


    normalize_after_mutation: bool = True
    current_generation: int = 0

    n_bw_iterations_per_gen: int = 0
    n_bw_iterations_after_ga: int = 0
    n_bw_iterations_before_ga: int = 0
    apply_bw_every_nth_gen: int = 1
    



    def __init__(
        self,
        n_symbols: int,
        n_states: int,
        population_size: int,
        n_generations: int,
        observations: MultipleObservationSequences,
        param_generator_func: ParamGeneratorFunction2 = uniform_random_left_right_hmm_params
        ) -> None:

        # Parametrized Attributes
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.population_size = population_size
        self.n_generations = n_generations
        self.observations = observations
        self.param_generator_func = staticmethod(param_generator_func)

        # Defaults for not Parametrized Attributes
        self.parent_pool_size = population_size

        self.ranges = representation.calc_chromosome_ranges(n_states, n_symbols)
        self.slices = representation.calc_chromosome_slices(n_states, n_symbols)
        self.n_genes = representation.calc_chromosome_length(n_states, n_symbols)

        self.row_stochastic_cutpoints = numpy.concatenate((
            numpy.arange(*self.ranges.B),
            numpy.arange(*self.ranges.A)
        ))

        self.population = create_population(population_size, n_states, n_symbols, param_generator_func)
        
        first_chromosome = self.population[0]
        self.mask = representation.calculate_chromosome_mask(first_chromosome)

        # self.performance_timers = {
        #     'mutation': PerformanceTimer(),
        #     'crossover': PerformanceTimer(),
        #     'fitness': PerformanceTimer(),
        #     'selection': PerformanceTimer(),
        #     'total': PerformanceTimer()
        # }

    # def _initialize_population(self):
    #     hmm_params_list = [self.param_generator_func(self.n_states, self.n_symbols) for i in range(self.population_size)]
    #     multiple_hmm_params = representation.hmm_params_list_as_multiple_hmm_params(hmm_params_list)
    #     self.population = representation.multiple_hmm_params_as_population(multiple_hmm_params)


    # def _train_population_with_bw(self, n_iterations=1):
    #     trained_population, fitness_values = train_population_with_bw()
        
    #     hmm_params = representation.population_as_multiple_hmm_params(self.population, self.n_states, self.n_symbols)
        
    #     reestimated_hmm_params, log_prob_traces = bw.train_multiple_hmms(hmm_params, self.observations, n_iterations)

    #     self.population = representation.multiple_hmm_params_as_population(reestimated_hmm_params)

    #     self.population[:, representation.FIELDS.FITNESS.start] = log_prob_traces[:, -1]
    #     self.logs.insert_bw_iterations(log_prob_traces)

    def _initialize_logs(self):
        n_log_entries = 0
        n_log_entries += self.n_bw_iterations_before_ga
        n_log_entries += self.n_bw_iterations_after_ga
        n_log_entries += self.n_generations
        n_log_entries += self.n_bw_iterations_per_gen * (self.n_generations // self.apply_bw_every_nth_gen)
        n_log_entries += 1 #Weil am Anfang einmal initial die Fitness berechnet wird

        self.logs = GABWLogger(self.population_size, n_log_entries)
    
    def _initialize_n_parents_and_n_children_per_generation(self):
        self.n_children_per_generation = self.population_size - self.keep_elitism
        self.n_matings_per_generation = self.n_children_per_generation // self.n_children_per_mating
        self.n_parents_per_generation = self.n_matings_per_generation * self.n_parents_per_mating

    
        # chromosomes[:, representation.FIELDS.FITNESS] = fitness_column

    
    # def sort_population_by_fitness(self):
    #     sorted_indices = self.population[:, representation.FIELDS.FITNESS.start].argsort()
    #     reverse_sorted_indices = numpy.flip(sorted_indices)
    #     self.population[:, :] = self.population[reverse_sorted_indices]
        
    #     ranks = numpy.arange(self.population_size).reshape((self.population_size, 1))
    #     self.population[:, self.slices.rank] = ranks
    
    def do_selection_step(self):
        parent_pool = self.population[:self.parent_pool_size]
        parents = self.parent_select_func(parent_pool, self.n_parents_per_generation, self)
        return parents

    def do_crossover_step(self, parents: Population):

        n_parents, chromosome_length = parents.shape
        n_children = (n_parents // self.n_parents_per_mating) * self.n_children_per_mating

        children = numpy.zeros((n_children ,chromosome_length))

        childrens_index = 0
        parents_index = 0
        for i in range(n_children):
            parents_for_children = parents[parents_index:(parents_index + self.n_parents_per_mating)]
            children_of_parents = self.crossover_func(parents_for_children, self.n_children_per_mating, self)
            children[childrens_index:(childrens_index + self.n_children_per_mating)] = children_of_parents

            childrens_index += self.n_children_per_mating
            parents_index += self.n_parents_per_mating

        return children

    def do_mutation_step(self, children: Population) -> Population:
        n_children = children.shape[0]
        
        for i in range(n_children):
            mutated_child = self.mutation_func(children[i, :], self)
            children[i] = mutated_child
        
        return children

    def smooth_emission_probabilities(self, chromosomes: Population):
        chromosomes = numpy.atleast_2d(chromosomes)
        smoothing_value = 1e-10
        chromosomes[:, self.slices.B] += smoothing_value
        return chromosomes


    def do_replacement_step(self, children):
        self.population[self.keep_elitism:, :] = children


    def _start(self):
        # self.performance_timers['total'].start()
        for i in range(self.n_generations):
            # Selection
            # self.performance_timers['selection'].start()
            parents = self.do_selection_step()
            # self.performance_timers['selection'].stop()

            # Crossover
            # self.performance_timers['crossover'].start()
            children = self.do_crossover_step(parents)
            # self.performance_timers['crossover'].stop()

            # Mutation
            # self.performance_timers['mutation'].start()
            mutated_children = self.do_mutation_step(children)
            # self.performance_timers['mutation'].stop()

            # smoothed_children = self.smooth_emission_probabilities(mutated_children)
            # normalized_children = self.normalize_children(children)

            normalized_children = normalize_population(mutated_children)
            # normalized_children[:, representation.FIELDS.FITNESS] = self.calculate_fitness_values(normalized_children)
            # self.performance_timers['fitness'].start()
            normalized_children[:, representation.FIELDS.FITNESS] = calculate_fitness_of_population(normalized_children, self.observations)
            # self.performance_timers['fitness'].stop()

            self.do_replacement_step(normalized_children)
            population_fitness_values = self.population[:, representation.FIELDS.FITNESS]
            self.logs.insert_ga_iterations(population_fitness_values)
            

            if (self.n_bw_iterations_per_gen > 0) and  (i % self.apply_bw_every_nth_gen == 0):
                self.population = train_population_with_bw(self.population, self.observations, self.n_bw_iterations_per_gen)

            self.population = sort_population_by_fitness_values(self.population)
        
        # self.performance_timers['total'].stop()

    def bake(self):
        self._initialize_logs()
        self._initialize_n_parents_and_n_children_per_generation()

    def start(self):
        # These attributes can't be initialized in __init__
        # because the values are not set via the constructor
        self.bake()

        self.population[:, representation.FIELDS.FITNESS] = calculate_fitness_of_population(self.population, self.observations)
        population_fitness_values = self.population[:, representation.FIELDS.FITNESS]
        self.logs.insert_ga_iterations(population_fitness_values)
        
        if self.n_bw_iterations_before_ga > 0:
            self.population = train_population_with_bw(self.population, self.observations, self.n_bw_iterations_before_ga)
        
        self.population = sort_population_by_fitness_values(self.population)
        self._start()

        best_chromosome_hmm = representation.chromosome_as_hmm_params(self.population[0])
        best_chromosome_fitness = self.population[0, representation.FIELDS.FITNESS]
        if self.n_bw_iterations_after_ga > 0:
            
            best_chromosome_hmm, log_trace = bw.train_single_hmm(best_chromosome_hmm, self.observations, self.n_bw_iterations_after_ga)
            best_chromosome_fitness = log_trace[-1]

        return best_chromosome_hmm, best_chromosome_fitness

    def plot(self):
        self.logs.plot()



