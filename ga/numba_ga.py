from lib.utils import rand_stochastic_vector, rand_stochastic_matrix, normalize_matrix, normalize_vector
import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Tuple, NamedTuple
from hmm.hmm import random_left_right_hmm_params, ParamGeneratorFunction2, MultipleObservationSequences
import lib.utils as utils
from numba import jit, njit
from numba.experimental import jitclass
import hmm.bw as bw
from ga.gabw_logger import GABWLogger

# from hmm.bw_numba import train




import seaborn as sns


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

spec = [

]



class GaHMM:
    param_generator_func: ParamGeneratorFunction2 = staticmethod(random_left_right_hmm_params)
    fitness_func: FitnessFunction = None
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
    apply_bw_every_nth_generaton: int = 1


    _n_iterations = 0


    def __init__(
        self,
        n_symbols: int,
        n_states: int,
        population_size: int,
        n_generations: int,
        observations: MultipleObservationSequences
        ) -> None:

        # Parametrized Attributes
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.population_size = population_size
        self.n_generations = n_generations
        self.observations = observations

        self._initialize_ranges_and_slices()
        self._initialize_n_genes()
        self._initialize_population()
        self._initialize_population_fitness_and_rank()
        self._initialize_hmms()
        self._initialize_masks()
    
    def _initialize_ranges_and_slices(self):
        start, stop, step = 0, self.n_states, self.n_states
        PI_range = RangeTuple(start, stop, step)
        PI_slice = slice(start, stop)

        start, stop, step = PI_range.stop, (PI_range.stop + self.n_states * self.n_symbols), self.n_states
        B_range = RangeTuple(start, stop, step)
        B_slice = slice(start, stop)

        start, stop, step = B_range.stop, (B_range.stop + self.n_states**2), self.n_states
        A_range = RangeTuple(start, stop, step)
        A_slice = slice(start, stop)

        fitness_range = RangeTuple(start=A_range.stop, stop=A_range.stop + 1, step=1)
        fitness_slice = slice(fitness_range.start, fitness_range.stop)

        rank_range = RangeTuple(start=fitness_range.stop, stop=fitness_range.stop + 1, step=1)
        rank_slice = slice(rank_range.start, rank_range.stop)

        self.ranges = ChromosomeRanges(PI_range, B_range, A_range, fitness_range, rank_range)
        self.slices = ChromosomeSlices(PI_slice, B_slice, A_slice, fitness_slice, rank_slice)
    
    def _initialize_n_genes(self):
        self.n_genes = self.ranges[-1].stop


    def _initialize_population(self):
        population = numpy.zeros((self.population_size, self.n_genes))

        for i in range(self.population_size):
            hmm_params = self.param_generator_func(self.n_states, self.n_symbols)
            PI, B, A = hmm_params

            population[i, self.slices.PI] = PI
            population[i, self.slices.B] = B.flatten()
            population[i, self.slices.A] = A.flatten()

        self.population = population

    def _initialize_population_fitness_and_rank(self):
        self.population[:, self.slices.fitness] = float('-inf')

        ranks_shape = (self.population_size, 1)
        default_rank_values = numpy.arange(self.population_size)
        self.population[:, self.slices.rank] = default_rank_values.reshape(ranks_shape)

    def _initialize_hmms(self):
        # Wichtig die hmms sind eine view der Population
        # Wenn man also hmms verändert oder chromosome verändert wirkt sich das auf der anderen seite aus

        PIs = self.population[:, self.slices.PI]
        
        Bs_shape = (self.population_size, self.n_states, self.n_symbols)
        Bs = self.population[:, self.slices.B].reshape(Bs_shape)

        As_shape = (self.population_size, self.n_states, self.n_states)
        As = self.population[:, self.slices.A].reshape(As_shape)

        self.hmms = MultipleHmmParams(PIs, Bs, As)

    def _initialize_masks(self):
        genes = self.population[0, :]
        masked_genes = numpy.ma.masked_where((genes == 0) | (genes == 1), genes)
        mask = masked_genes.mask
        mask[self.slices.rank] = True
        mask[self.slices.fitness] = True
        self.mask = mask


    def _train_with_bw_for_n_iterations(self, n_iterations=1):
        
        
        reestimated_hmm_params, log_prob_traces = bw.train_multiple_hmms(self.hmms, self.observations, n_iterations)
        PIs, Bs, As = reestimated_hmm_params

        # the values can only be assigned via slicing because we don't want to loose the reference to the population array
        self.hmms.PIs[:, :] = PIs
        self.hmms.Bs[:, :, :] = Bs
        self.hmms.As[:, :, :] = As

        self.logs.insert_bw_iterations(log_prob_traces)

    def _initialize_logs(self):
        n_log_entries = 0
        n_log_entries += self.n_bw_iterations_before_ga
        n_log_entries += self.n_bw_iterations_after_ga
        n_log_entries += self.n_generations
        n_log_entries += self.n_bw_iterations_per_gen * (self.n_generations // self.apply_bw_every_nth_generaton)

        self.logs = GABWLogger(self.population_size, n_log_entries)
    
    def _initialize_n_parents_and_n_children_per_generation(self):
        self.n_children_per_generation = self.population_size - self.keep_elitism
        self.n_matings_per_generation = self.n_children_per_generation // self.n_children_per_mating
        self.n_parents_per_generation = self.n_matings_per_generation * self.n_parents_per_mating

    def calculate_fitness_values(self):
        fitness_values = bw.calc_total_log_prob_for_multiple_hmms(self.hmms, self.observations)
        fitness_shape = (self.population_size, 1)
        fitness_column = fitness_values.reshape(fitness_shape)
        self.population[:, self.slices.fitness] = fitness_column

        self.logs.insert_ga_iterations(fitness_column)
    
    def sort_population_by_fitness(self):
        sorted_indices = self.population[:, self.slices.fitness.start].argsort()
        reverse_sorted_indices = numpy.flip(sorted_indices)
        self.population[:, :] = self.population[reverse_sorted_indices]
        
        ranks = numpy.arange(self.population_size).reshape((self.population_size, 1))
        self.population[:, self.slices.rank] = ranks
    
    def do_selection_step(self):
        parents = self.parent_select_func(self.population, self.n_parents_per_generation, self)
        return parents

    def do_crossover_step(self, parents):

        children = numpy.zeros((self.n_children_per_generation ,self.n_genes))
        child_index = 0

        for parent_index in range(0, self.n_parents_per_generation, self.n_parents_per_mating):
            
            par = parents[parent_index : (parent_index+ self.n_parents_per_mating), :].copy()
            childs = self.crossover_func(par, self.n_children_per_mating, self)

            children[child_index:(child_index + len(childs)), : ] = childs

            child_index +=len(childs)

        
        if not child_index == len(children): raise Exception( "The Crossover Function does not support the provided n_children_per_mating" )

        return children

    def do_mutation_step(self, children):
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

    def normalize_population(self):
        PIs_normalization = numpy.atleast_2d(self.hmms.PIs.sum(axis=1)).T
        self.hmms.PIs[:,:] = self.hmms.PIs / PIs_normalization

        Bs_normalization = numpy.atleast_3d(self.hmms.Bs.sum(axis=2))
        self.hmms.Bs[:,:] = self.hmms.Bs / Bs_normalization

        As_normalization = numpy.atleast_3d(self.hmms.As.sum(axis=2))
        self.hmms.As[:,:] = self.hmms.As / As_normalization


    

    def do_replacement_step(self, children):
        self.population[self.keep_elitism:, :] = children


    def _start(self):
        for i in range(self.n_generations):
            
            
            parents = self.do_selection_step()
            children = self.do_crossover_step(parents)
            mutated_children = self.do_mutation_step(children)
            
            smoothed_children = self.smooth_emission_probabilities(mutated_children)
            # normalized_children = self.normalize_children(children)

            self.do_replacement_step(smoothed_children)
            self.normalize_population()

            self.calculate_fitness_values()
            self.sort_population_by_fitness()
            


    def bake(self):
        self._initialize_logs()
        self._initialize_n_parents_and_n_children_per_generation()

    def start(self):
        # These attributes can't be initialized in __init__
        # because the values are not set via the constructor
        self.bake()
        
        self._train_with_bw_for_n_iterations(self.n_bw_iterations_before_ga)
        self._start()
        self._train_with_bw_for_n_iterations(self.n_bw_iterations_after_ga)
        # call gabw before

        # start

        # call gabw after
    
    


    
    
    def plot(self):
        self.logs.plot()


        # # Ab hier beginnt der Müll:
        # self.n_genes = self.calc_n_genes(n_states, n_symbols)
        # self.slices = self.calculate_slices(self.n_states, self.n_symbols)
        # self.population = self.initialize_population()
        # # Assign Ranks to have default values for selection function
        # self.assign_ranks_to_population()
        
        # self.chromosome_mask = self.initialize_chromosome_mask()

        # self.calc_n_parents_and_children_per_generation()
































    
    
    


    

    


    # def calc_n_parents_and_children_per_generation(self):
    #     self.n_children_per_generation = self.population_size - self.keep_elitism
    #     self.n_matings_per_generation = self.n_children_per_generation // self.n_children_per_mating
    #     self.n_parents_per_generation = self.n_matings_per_generation * self.n_parents_per_mating

    # def _validate(self):

    #     children_per_generation_is_multiple_of_children_per_mating = self.n_children_per_generation % self.n_children_per_mating == 0
    #     if not children_per_generation_is_multiple_of_children_per_mating:
    #         raise ValueError("Number of Children per Generation must be divisible by Number of Childrens Per Mating")


    # def initialize_masks(self) -> ChromosomeMask:
    #     start_vector = self.hmms.start_vectors[0]
    #     masked_start_vector = numpy.ma.masked_where((start_vector == 0) | (start_vector == 1), start_vector)

    #     emission_matrix = self.hmms.emission_matrices[0]
    #     masked_emission_matrix = numpy.ma.masked_where((emission_matrix == 0) | (emission_matrix == 1), emission_matrix)

    #     transition_matrix = self.hmms.transition_matrices[0]
    #     masked_transition_matrix = numpy.ma.masked_where((transition_matrix == 0) | (transition_matrix) == 1, transition_matrix)

    #     masks = HmmParams(
    #         masked_start_vector.mask,
    #         masked_emission_matrix.mask,
    #         masked_transition_matrix.mask
    #     )

    #     self.masks = masks



    # def order_and_rank_population(self):
    #     # self.order[n] is the index of the nth fittest chromosome
    #     # self.rank[n] is the rank of the nth chromosome
    #     self.order = numpy.flip(self.fitness.argsort())
    #     self.rank = self.order.argsort()

    
    

    

    
    # def initialize_logs(self):
    #     return numpy.ndarray((self.population_size, self.n_generations + 1))

    
    

    # def normalize_chromosomes(self):
    #     start_vectors_sum = numpy.atleast_2d(self.hmms.start_vectors.sum(axis=1))
    #     self.hmms.start_vectors[:,:] = self.hmms.start_vectors / start_vectors_sum

    #     emission_probs_sum = numpy.atleast_3d(self.hmms.emission_matrices.sum(axis=2))
    #     self.hmms.emission_matrices[:,:,:] = self.hmms.emission_matrices / emission_probs_sum

    #     transition_probs_sum = numpy.atleast_3d(self.hmms.transition_matrices.sum(axis=2))
    #     self.hmms.transition_matrices[:,:,:] = self.hmms.transition_matrices / transition_probs_sum


    
    # def do_selection_step(self):
        
    #     parents = self.parent_select_func(self.population, self.n_parents_per_generation, self.slices, self)
    #     return parents.copy()

    # def do_crossover_step(self, parents):

    #     children = numpy.zeros((self.n_children_per_generation ,self.n_genes))
    #     child_index = 0

    #     for parent_index in range(0, self.n_parents_per_generation, self.n_parents_per_mating):
            
    #         par = parents[parent_index : (parent_index+ self.n_parents_per_mating), :].copy()
    #         childs = self.crossover_func(par, self.n_children_per_mating, self.slices, self)

    #         children[child_index:(child_index + len(childs)), : ] = childs

    #         child_index +=len(childs)

        
    #     if not child_index == len(children): raise Exception( "The Crossover Function does not support the provided n_children_per_mating" )


    #     children = self.assign_default_values_to_hidden_genes(children)

    #     return children

    # def assign_default_values_to_hidden_genes(self, population):
    #     population[:, self.slices.rank.start] = 0
    #     population[:, self.slices.fitness.start]= float('-inf')
    #     return population
        

    # def do_mutation_step(self, children):
    #     n_children = children.shape[0]
        
    #     for i in range(n_children):
    #         mutated_child = self.mutation_func(children[i, :], self.slices, self.chromosome_mask, self)
    #         children[i] = mutated_child.copy()
        
    #     return children.copy()

    # def smooth_emission_probabilities(self):
    #     # Emission values can't be zero otherwise Baum-Welch doesn't work
    #     smoothing_value = 1e-10
    #     self.hmms.emission_matrices[:,:,:] += smoothing_value
    #     start, stop, _ = self.slices.emission_probs
    #     self.population[:, start:stop] = self.population[:, start: stop] + smoothing_value


    # def _do_ga_iteration(self):
    #     self.calculate_fitness()
    #     self.sort_population()
    #     self.assign_ranks_to_population()
    #     fitness_values = self.population[self.slices.fitness.start]
    #     self.update_logs(fitness_values)


    #     parents = self.do_selection_step()
    #     children_after_cross = self.do_crossover_step(parents)
    #     children = self.do_mutation_step(children_after_cross.copy())


    #     self.population[self.keep_elitism:, :] = children
    #     self.smooth_emission_probabilities()
    #     self.normalize_chromosomes()


    


    # def _start(self):
    #     # print(f'starting iteration {iteration}')
    #     # We can't start at zero because the population might have been trained with bw before
    #     for i in range(self.n_generations):

    #         self._do_ga_iteration()

    #         apply_bw = (self.n_bw_iterations_per_gen > 0) and  (i % self.apply_bw_every_nth_generaton)
    #         if apply_bw:
    #             self._train_with_bw_for_n_iterations(self.n_bw_iterations_per_gen)

            

    #         # self.current_generation should be equal to the number of 
    #         # self.current_generation += 1

    


    # def start(self):
    #     self._n_iterations = self._calc_n_log_entries()
    #     self.logs = self.initialize_logs()
    #     self._validate()


    #     self._train_with_bw_for_n_iterations(self.n_bw_iterations_before_ga)
    #     self._start()
    #     self._train_with_bw_for_n_iterations(self.n_bw_iterations_after_ga)


        


    # def plot(self):
    #     self.logs.plot()



