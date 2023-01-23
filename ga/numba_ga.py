from lib.utils import rand_stochastic_vector, rand_stochastic_matrix, normalize_matrix, normalize_vector
import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Tuple, NamedTuple
from hmm.hmm import random_left_right_hmm_params
import lib.utils as utils
from hmm.bw import BaumWelch
import copy
from numba import jit, njit
from hmm.bw_numba import HmmParams

class SliceTuple(NamedTuple):
    start: int
    stop: int
    step: int

class ChromosomeSlices(NamedTuple):
    start_probs: SliceTuple
    emission_probs: SliceTuple
    transition_probs: SliceTuple
    fitness: SliceTuple
    rank: SliceTuple







class Chromosome:
    def __init__(self, start_vector:numpy.ndarray, emission_matrix:numpy.ndarray, transition_matrix:numpy.ndarray):
        self.start_vector = start_vector
        self.emission_matrix = emission_matrix
        self.transition_matrix = transition_matrix

        self.n_states = emission_matrix.shape[0] 
        self.n_symbols = emission_matrix.shape[1]

        self.n_genes = self.calc_n_genes()

        self.indices_range = {
            'S': (0, self.n_states, self.n_states),
            'E': (self.n_states, (self.n_states + self.n_states*self.n_symbols), self.n_symbols),
            'T': ((self.n_states + self.n_states*self.n_symbols), self.n_genes, self.n_states),
        }

        genes = self.create_genes_from_matrices()
        self.genes = self.mask_immutable_genes(genes)
        
        self.log_probability = float('-inf')
        self.fitness = 0
        self.rank = 0

    def calc_n_genes(self):
        return self.n_states + self.n_states*self.n_symbols + self.n_states**2
            
    def create_genes_from_matrices(self):
        genes = numpy.empty(self.n_genes)

        start_S, stop_S, _ = self.indices_range['S']
        start_E, stop_E, _ = self.indices_range['E']
        start_T, stop_T, _ = self.indices_range['T']

        genes[start_S:stop_S] = self.start_vector
        genes[start_E:stop_E] = self.emission_matrix.flatten()
        genes[start_T:stop_T] = self.transition_matrix.flatten()

        return genes
        
    def mask_immutable_genes(self, genes: numpy.ndarray) ->numpy.ma.masked_array:
        """Applies the mask if provided. Otherwise masks all indices where the value is equal to 0 or 1

        Args:
            genes (_type_): _description_
            mask (_type_): _description_

        Returns:
            _type_: _description_
        """

        return numpy.ma.masked_equal(numpy.logical_or(genes == 1, genes == 0), genes, copy=False)

    def normalize(self):
        self.start_vector = self.start_vector / numpy.sum(self.start_vector)
        self.transition_matrix = self.transition_matrix / numpy.sum(self.transition_matrix, axis=1, keepdims=True)
        self.emission_matrix = self.emission_matrix / numpy.sum(self.emission_matrix, axis=1, keepdims=True)

        self.genes = self.create_genes_from_matrices()

        
    def clone(self):
        
        clone = copy.copy(self)
        clone_genes = self.genes.copy()
        clone.genes = clone_genes

        return clone
        
    def __lt__(self, other: 'Chromosome'):
        return self.log_probability < other.log_probability
    
    def __gt__(self, other: 'Chromosome'):
        return self.log_probability > other.log_probability
    









class GaHMM:
    population: List[Chromosome] = []

    def __init__(
        self,
        n_symbols: int,
        n_states: int,
        population_size: int,
        n_generations: int,
        fitness_func: Callable[[Chromosome, 'GaHMM'], float],
        parent_select_func: Callable[[List[Chromosome], int, 'GaHMM'], List[List[Chromosome]]],
        mutation_func: Callable[[Chromosome], Chromosome], 
        crossover_func: Callable[[List[Chromosome]], Chromosome],
        keep_elitism=1,
        normalize_after_mutation=True,
        param_generator_func: Callable[[int, int], Tuple[numpy.array, numpy.ndarray, numpy.ndarray]] = random_left_right_hmm_params,
        ) -> None:

        # Parametrized Attributes
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.population_size = population_size
        self.n_generations = n_generations
        self.fitness_func = staticmethod(fitness_func)
        self.parent_select_func = staticmethod(parent_select_func)
        self.mutation_func = staticmethod(mutation_func)
        self.crossover_func = staticmethod(crossover_func)
        self.keep_elitism=keep_elitism
        self.normalize_after_mutation=normalize_after_mutation
        self.param_generator_func = staticmethod(param_generator_func)

        # Calculated Attributes
        self.offspring_count = self.population_size - self.keep_elitism
        self.current_generation = 0

        self.population = [self.new_chromosome() for i in range(self.population_size)]

        self.logs = {
            'total': [],
            'max': [],
            'min': [],
            'mean': [],
        }

    @staticmethod
    @njit
    def calculate_slices(n_states: int, n_symbols: int) -> ChromosomeSlices:
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

    @staticmethod
    @njit
    def initialize_population(
        slices: ChromosomeSlices,
        n_states: int,
        n_symbols: int,
        population_size: int, 
        param_generator_func: Callable[[int, int], Tuple[numpy.array, numpy.ndarray, numpy.ndarray]]):
        chromosome_length = slices.rank.stop

        population = numpy.zeros((population_size, chromosome_length))

        for i in range(population_size):
            start_vector, emission_matrix, transition_matrix = param_generator_func(n_states, n_symbols)
            start, stop, _ = slices.start_probs
            population[i, start: stop] = start_vector
            start, stop, _ = slices.emission_probs
            population[i, start: stop] = emission_matrix.flatten()
            start, stop, _ = slices.transition_probs
            population[i, start: stop] = transition_matrix.flatten()
        
        return population

    

    def calculate_fitness(population: numpy.ndarray, slices: ChromosomeSlices, fitness_func: Callable[[HmmParams], float]):
        population_size = population.shape[0]
        n_states = slices.transition_probs.step
        n_symbols = slices.emission_probs.step

        for i in range(population_size):
            start, stop, _ = slices.start_probs
            start_vector = population[i, start:stop]
            start, stop, _ = slices.emission_probs
            emission_matrix = population[i, start:stop].reshape((n_symbols, n_states))
            start, stop, _ = slices.transition_probs
            transition_matrix = population[i, start:stop].reshape((n_states, n_states))

            hmm_params = HmmParams(start_vector, emission_matrix, transition_matrix)

            log_prob = fitness_func(hmm_params)
            population[i, slices.fitness] = log_prob
    
    def update_fitness(self):
   
        total_probability = 0
        min_probability = float('inf')
        max_probability = float('-inf')
        prob_sum = 0
        
        for chromosome in self.population:
            log_prob = self.fitness_func(chromosome, self)
            chromosome.log_probability = log_prob
            total_probability += log_prob
            prob_sum += numpy.exp(log_prob)

            if log_prob < min_probability:
                min_probability = log_prob
            
            if log_prob > max_probability:
                max_probability = log_prob
        
        mean_probability = total_probability/self.population_size

        self.logs['max'].append(max_probability)
        self.logs['min'].append(min_probability)
        self.logs['mean'].append(mean_probability)
        self.logs['total'].append(total_probability)


        # Update Fitness and Rank
        self.population.sort(reverse=True)
        for i in range(self.population_size):
            chromosome = self.population[i]

            # numpy.exp(chromosome.probability) / prob_sum

            # chromosome.fitness = chromosome.probability/total_probability
            chromosome.rank = i


    def normalize_chromosome(self, chromosome: Chromosome):
        for i in range(len(self.legal_slice_points) - 1):
            start = self.legal_slice_points[i]
            stop = self.legal_slice_points[i+1]
            chromosome.genes[start:stop] = normalize_vector(chromosome.genes[start:stop])
        
        return chromosome

    def start(self):
        for iteration in self.n_generations:
            self.update_fitness()

            parent_pairs = self.parent_select_func(self.population, self.offspring_count, self)
            offspring = []
            for parents in parent_pairs:
                children = self.crossover_func(parents, self)
                offspring += children

            if len(offspring) != self.offspring_count:
                raise ValueError(f'The number of offspring after crossover and mutation is {len(offspring)} and does not match the expected Value of {self.offspring_count}')

            offspring = [self.mutation_func(c, self) for c in offspring]


            if self.normalize_after_mutation:
                for chromosome in offspring:
                    chromosome.normalize()


            # neue population ist keep elitism + offspring
            elites = self.population[:self.keep_elitism]
            self.population = elites + offspring
            

        return self.logs

    def plot(self):
        x = range(self.n_generations)
        
        plt.plot(x, self.logs['max'], label='max')
        plt.plot(x, self.logs['mean'], label='mean')
        plt.plot(x, self.logs['min'], label='min')

        plt.legend()
        plt.show()


    def new_chromosome(self) -> Chromosome:
        S, E, T = self.param_generator_func(self.n_states, self.n_symbols)
        return Chromosome(S, E, T)

