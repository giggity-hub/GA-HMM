from lib.utils import rand_stochastic_vector, rand_stochastic_matrix
import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, List, Dict


# the random genes should not be initialized inside the individual because that would limit the options
# or cause to many parameters to be needed


def genes_from_params(start_probs, transition_probs, emission_probs):
    [start_probs] + transition_probs + emission_probs


def slice_genes(genes: List[float], n_states: int, n_symbols: int):
    # genes = [ start_probs | emission_probs     | transition_probs  ]
    # genes = [ n_states    | n_states*n_symbols | n_states*n_states ]
    slice_start_probs = slice(0, n_states)
    slice_emission_probs = slice(n_states, (n_states + n_states*n_symbols))
    slice_transition_probs = slice(-n_states*n_states, len(genes))

    start_probs = genes[slice_start_probs]
    emission_probs = genes[slice_emission_probs]
    transition_probs = genes[slice_transition_probs]

    return start_probs, emission_probs, transition_probs

def normalize_genes(genes: List[float], n_states: int, n_symbols: int):
    start_probs, emission_probs, transition_probs = slice_genes(genes, n_states, n_symbols)

    # normalize matrixs

class Individual:

    def __init__(self, 
        n_states: int, 
        alphabet: List[str],
        genes:List =[float]) -> None:
        """_summary_

        Args:
            n_states (int): _description_
            alphabet (List[str]): _description_
            genes (List[List[float]], optional): _description_. Defaults to [].
        """
       
        self.n_symbols: int = len(alphabet)
        self.n_states: int = n_states
        self.alphabet: List[str] = alphabet

        self.fitness: float = float('-inf')

        # genes müssen übergeben werden, weil sonst der lachs mit dem crossover nicht so kurz programmierbar ist

        # if not genes:
        #     start_probs = rand_stochastic_vector(n_states)
        #     transition_probs = rand_stochastic_matrix(n_states, n_states)
        #     emission_probs = rand_stochastic_matrix(n_states, self.n_symbols)
        #     genes = [start_probs] + transition_probs + emission_probs
        self.genes = genes

    def to_hmm(self):
        emmission_probs_dicts = [dict(zip(self.alphabet, probs)) for probs in self.emission_probs]
        emmission_probs_dists = [DiscreteDistribution(probs) for probs in emmission_probs_dicts]

        model = HiddenMarkovModel.from_matrix(
            self.transition_probs, 
            emmission_probs_dists, 
            self.start_probs
        )
        return model

    def get_state_emissions(self, state_index):
        return self.genes[(state_index +1)]
    def set_state_emissions(self, state_index, value):
        # +1 ist das offset weil die bei 1 anfangen
        self.genes[(state_index +1)] = value
    
    def get_state_transitions(self, state_index):
        return self.genes[(state_index + 1 + self.n_states)]
    
    def set_state_transitions(self, state_index, value):
        self.genes[(state_index + 1 + self.n_states)] = value


    @property
    def start_probs(self):
        return self.genes[0]

    @start_probs.setter
    def transition_probs(self, value):
        self.genes[0] = value

    @property
    def transition_probs(self):
        return self.genes[1: self.n_states+1]

    @transition_probs.setter
    def transition_probs(self, value):
        self.genes[1:self.n_states+1] = value 
    
    @property
    def emission_probs(self):
        return self.genes[self.n_states+1:]

    @emission_probs.setter 
    def emission_probs(self, value):
        self.genes[self.n_states+1:] = value

    




# Problem:
# If you want to base the parent selection as well as the mutation on the fitness
# then you need to call the fitness function twice per iteration.
# which is probably not so nice because the fitness function is pretty expensive

# Type Declarations


class GeneticAlgorithm:
    def __init__(self, 
            initial_population, 
            n_generations, 
            fitness_func: Callable[[Individual, 'GeneticAlgorithm'], float], 
            parent_select_func, 
            crossover_func, 
            mutation_func, 
            keep_elitism=1 ) -> None:
        """_summary_

        Args:
            initial_population (_type_): _description_
            n_generations (_type_): _description_
            fitness_func (_type_): sheeeesh boiiiiii
            parent_select_func (_type_): _description_
            crossover_func (_type_): _description_
            mutation_func (_type_): _description_
            keep_elitism (int, optional): _description_. Defaults to 1.
        """
            

        self.population = initial_population
        self.population_size = len(initial_population)
        self.n_generations = n_generations
        self.current_generation = 0
        self.fitness_func = fitness_func
        self.parent_select_func = parent_select_func
        self.crossover_func = crossover_func
        self.keep_elitism = keep_elitism


        # The number of pairs of parents per generation. (! A pair of parents can consist of more than two)
        self.parent_count = self.population_size - self.keep_elitism


        # Output
        self.result = {
            'min': [],
            'max': [],
            'mean': []
        }

        # keep track of the average fitness, min and max fitness
    
    def plot(self):
        print(self.result['max'])
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(range(self.n_generations), self.result['max']);  # Plot some data on the axes.


    def start(self):
        # while self.current_generation < self.n_generations
        while self.current_generation < self.n_generations:

            # calculate fitness
            for individual in self.population:
                individual.fitness = self.fitness_func(individual, self)
                # track if fitness bigger than max or whatever
                
            # sort individuals by fitness
            self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)

            # update result
            fitness_max = self.population[0].fitness
            fitness_min = self.population[-1].fitness
            fitness_sum = sum(x.fitness for x in self.population)
            fitness_mean = fitness_sum / self.population_size
            self.result['max'].append(fitness_max)
            self.result['min'].append(fitness_min)
            self.result['mean'].append(fitness_mean)




            # children = population_size - keep_elitism
            # länge der parents muss = anzahl der offspring sein

            all_parents = self.parent_select_func(self.population, self.parent_count, self)
            if len(all_parents) != self.parent_count:
                raise Exception("The number of parents does not match the expected number")

            offspring = [self.crossover_func(parents, self) for parents in all_parents]
            

            # neue population ist keep elitism + offspring
            elites = self.population[:self.keep_elitism]
            self.population = elites + offspring
            

            # 5 paare von parents
            # jedes macht 2 kinder
            # 10 kinder


            # select parents for mating
            # > liste von partnern

            # crossover
            # for every pair of parents => crossover

            # children = summe aus allen crossover schmossovers


            # do mutation

            # neue generation builden:
            # 
            print(f'generation {self.current_generation} of {self.n_generations}')
            self.current_generation+=1
        return self.result

