from lib.utils import rand_stochastic_vector, rand_stochastic_matrix, normalize_matrix, normalize_vector
import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Tuple
from hmm.hmm import random_left_right_hmm_params
import lib.utils as utils




# child A
# Child B

# How to create a child without the ga instance
# child = parentA.copy()
# child.genes[x:y] = parent[1].genes[x:y]

# ChromosomeFactory

# createfromHMM
# createRandom
# createFromGenes


class Moped:
    def __init__(self, genes: numpy.ndarray, n_symbols, n_states):
        self.n_symbols = n_symbols
        self.n_states = n_states
        self.genes = genes

        self.range = {
            'S': (0, n_states, n_states),
            'E': (n_states, (n_states + n_states*n_symbols), n_symbols),
            'T': ((n_states + n_states*n_symbols), self.n_genes, n_states)
        }

        self.slice = {
            'S': numpy.s_[self.range['S'][0]: self.range['S'][1]],
            'E': numpy.s_[self.range['E'][0]: self.range['E'][1]],
            'T': numpy.s_[self.range['T'][0]: self.range['T'][1]]
        }

        self.S = self.genes[self.slice['S']]
        self.E = self.genes[self.slice['E']].reshape((n_states, n_symbols))
        self.T = self.genes[self.slice['T']].reshape((n_states, n_states))


    def forward(self, O: numpy.ndarray, ):
        # Wichtig Zeit ist auf der Y achse und State auf der X achse (damit multiplikation einfacher wird)
        alpha_shape = (len(observation_seq), self.n_states)
        alpha = numpy.zeros(alpha_shape)

        # Initialize Alpha with the starting probabilities times first observation symbol
        first_observation = observation_seq[0]
        alpha[0, :] = self.S * self.E[:, first_observation]

        for t in range(1, len(O)):
            for j in range(self.n_states):
                

                alpha[t, j] = moped * self.E[j, O[t]]


                alpha[t, :] = sum(alpha[t - 1] * self.T[j, :]) * self.E[:, observation_seq[t]]

            
                # Matrix Computation Steps
                #                  ((1x2) . (1x2))      *     (1)
                #                        (1)            *     (1)
                alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

                
 
        return alpha

    def transition_prob(self, from_state: int, to_state: int):
        pass



class ChromosomeSchmomosome:
    genes: numpy.array
    probability: float
    fitness: float
    rank: int
    def __init__(self, states_count: int, symbols_count: int, genes: numpy.array) -> None:
        self.genes: numpy.array = genes
        self.probability: float = float('-inf')
        self.fitness: float = 0
        self.rank: int = 0
        self.states_count = states_count 
        self.symbols_count = symbols_count
    
    def __lt__(self, other):
        return self.probability < other.probability
    
    def __gt__(self, other):
        return self.probability > other.probability


    def forward(observation_sequence: numpy.ndarray, time_index: int, final_state_index: int):
        pass

    def calculate_forward_trellis(self, observation_sequence: numpy.ndarray):
        observation_sequence_length = observation_sequence.shape[1]
        alpha = numpy.zeros((self.states_count, observation_sequence_length))

        alpha[0, :] = 

    def b(self, state_index, observation_index):
        
        emission_probability_matrix = self.genes[]

    

        # callculate initial probabilities

    def single_observation_log_prob(self, observation_sequence: numpy.ndarray):
        total = 0
        for i in range(self.states_count):
            total += self.forward(
                observation_sequence=observation_sequence,
                time_index=
                )


class Chromosome:
    genes: numpy.array
    probability: float
    fitness: float
    rank: int
    def __init__(self, genes: numpy.array) -> None:
        self.genes: numpy.array = genes
        self.probability: float = float('-inf')
        self.fitness: float = 0
        self.rank: int = 0
    
    def __lt__(self, other):
        return self.probability < other.probability
    
    def __gt__(self, other):
        return self.probability > other.probability










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
        self.n_genes = n_states + n_states**2 + n_states*n_symbols
        self.alphabet = [i for i in range(n_symbols)]
        self.offspring_count = self.population_size - self.keep_elitism
        self.current_generation = 0

        self.range = {
            'S': (0, n_states, n_states),
            'E': (n_states, (n_states + n_states*n_symbols), n_symbols),
            'T': ((n_states + n_states*n_symbols), self.n_genes, n_states)}

        self.len = {
            'S': n_states,
            'E': n_states*n_symbols,
            'T': n_states**2,
            'ET': n_states*n_symbols + n_states**2 }

        self.slice = {
            'S': numpy.s_[self.range['S'][0]: self.range['S'][1]],
            'E': numpy.s_[self.range['E'][0]: self.range['E'][1]],
            'T': numpy.s_[self.range['T'][0]: self.range['T'][1]],}

        self.shape = {
            'S': (self.n_states,),
            'E': (self.n_states, self.n_symbols),
            'T': (self.n_states, self.n_states)}

        self.logs = {
            'total': [],
            'max': [],
            'min': [],
            'mean': [],
        }

        self.legal_slice_points: List[int] = [i for X in ['S', 'E', 'T'] for i in range(*self.range[X])] + [self.n_genes]
        self.population = [self.new_chromosome() for i in range(self.population_size)]

    
    def update_fitness(self):
        total_probability = 0
        min_probability = float('inf')
        max_probability = float('-inf')
        prob_sum = 0
        
        for chromosome in self.population:
            log_prob = self.fitness_func(chromosome, self)
            chromosome.probability = log_prob
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

            numpy.exp(chromosome.probability) / prob_sum

            chromosome.fitness = chromosome.probability/total_probability
            chromosome.rank = i


    def normalize_chromosome(self, chromosome: Chromosome):
        for i in range(len(self.legal_slice_points) - 1):
            start = self.legal_slice_points[i]
            stop = self.legal_slice_points[i+1]
            chromosome.genes[start:stop] = normalize_vector(chromosome.genes[start:stop])
        
        return chromosome

    def start(self):
        while self.current_generation < self.n_generations:
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
                    self.normalize_chromosome(chromosome)

            # neue population ist keep elitism + offspring
            elites = self.population[:self.keep_elitism]
            self.population = elites + offspring
            
            self.current_generation+=1
        return self.logs

    def plot(self):
        x = range(self.n_generations)
        
        plt.plot(x, self.logs['max'], label='max')
        plt.plot(x, self.logs['mean'], label='mean')
        plt.plot(x, self.logs['min'], label='min')

        plt.legend()
        plt.show()



    def _hmm_from_params(self, S: numpy.array, E: numpy.ndarray, T: numpy.ndarray):
        distributions = []
        for row in E:
            emission_dict = dict(zip(self.alphabet, row))
            emission_dist = DiscreteDistribution(emission_dict)
            distributions.append(emission_dist)
        
        model = HiddenMarkovModel.from_matrix(
            T, 
            distributions, 
            S
        )
        return model

    def new_chromosome(self) -> Chromosome:
        S, E, T = self.param_generator_func(self.n_states, self.n_symbols)
        genes = numpy.empty(self.n_genes)

        genes[self.slice['S']] = S
        genes[self.slice['E']] = E.flatten()
        genes[self.slice['T']] = T.flatten()

        return Chromosome(genes)

    def new_hmm(self):
        S, E, T = self.param_generator_func(self.n_states, self.n_symbols)
        hmm =  self._hmm_from_params(S, E, T)
        return hmm

    def chromosome2hmm(self, chromosome: Chromosome):

        S = chromosome.genes[self.slice['S']]
        E = chromosome.genes[self.slice['E']].reshape(self.shape['E'])
        T = chromosome.genes[self.slice['T']].reshape(self.shape['T'])

        model = self._hmm_from_params(S, E, T)
        return model

    def hmm2chromosome(self, hmm: HiddenMarkovModel) -> Chromosome:
         # Step 1: Get that motherfucking transitions shit
        transition_matrix: numpy.matrix = hmm.dense_transition_matrix()

        # The Start Probs are the outgoing transitions from the silent start state
        S = transition_matrix[hmm.start_index, :hmm.silent_start]

        # Remove Silent States
        T = transition_matrix[:hmm.silent_start, :hmm.silent_start]


        n = self.shape['E'][0] * self.shape['E'][1]
        E = numpy.empty(n)
        step = self.shape['E'][1]

        for i in range(self.n_states):
            state = hmm.states[i]
            # Reminder: We don't check whether the values are in the correct order (they better be otherwise this won't work)
            E[i: i+step] = state.distribution.values()


        genes = numpy.empty(self.n_genes)
        genes[self.slice['S']] = S
        genes[self.slice['E']] = E.flatten()
        genes[self.slice['T']] = T.flatten()

        return Chromosome(genes)

