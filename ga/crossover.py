import numpy
from typing import List
from ga.ga import Individual, GeneticAlgorithm
import random
import numpy.random as npr

# def single_row_cutpoint(cut_points):
#     # cut_points = get_cut_points(num_states, num_symbols)

#     def crossover_func(parents, offspring_size, ga_instance):
#         num_offspring = offspring_size[0]
#         offspring = []
#         idx = 0

#         while len(offspring) < num_offspring:
#             parent1 = parents[idx % parents.shape[0], :].copy()
#             parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

#             random_split_point = numpy.random.choice(cut_points)

#             parent1[random_split_point:] = parent2[random_split_point:]

#             offspring.append(parent1)

#             idx += 1

#         return numpy.array(offspring)

#     return crossover_func


def mean_crossover(parents: List[Individual] , ga_instance: GeneticAlgorithm):
    child_genes = (parents[0].genes + parents[1].genes) / 2

    child = Individual(
        genes=child_genes,
        normalize_genes=True
    )

    return child

def single_point_crossover(parents: List[Individual], ga_instance: GeneticAlgorithm):
    # how many crossover points are there?
    # [start] [transition] [emission]

    genes1 = parents[0].genes 
    genes2 = parents[1].genes

    crossover_index = random.randrange(0, len(genes1))

    child_genes = numpy.concatenate((
        genes1[0:crossover_index],
        genes2[crossover_index:]
    ))

    child = Individual(
        genes=child_genes, 
        normalize_genes=True
    )

    return child

def n_point_crossover(n, crossover_rate):

    

    def crossover(parents:List[Individual], ga_instance: GeneticAlgorithm):
        
        no_crossover = random.uniform(0,1) > crossover_rate
        if no_crossover:
            return parents

        # one crossover in transition matrix 

        tp_a = parents[0].transition_probs
        tp_b = parents[1].transition_probs

        ep_a = parents[0].emission_probs
        ep_b = parents[1].emission_probs


        # + transition offset
        tp_crossover = numpy.random.randint()
        # + emission offset
        ep_crossovers = sorted([npr.choice(len(ep_a)) for i in range(2)])

        slice_points = [0, n, ]

        random.uniform()

        # double crossover in 
        return parents

    return crossover