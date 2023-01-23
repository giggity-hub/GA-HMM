import numpy
from typing import List
from ga.ga import Chromosome, GaHMM
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


# def mean_crossover(parents: List[Individual] , ga_instance: GeneticAlgorithm):
#     child_genes = (parents[0].genes + parents[1].genes) / 2

#     child = Individual(
#         genes=child_genes,
#         normalize_genes=True
#     )

#     return [child]




def single_point_crossover(parents: List[Chromosome], gabw: GaHMM=None):

    parent_a, parent_b = parents
    
    lo, hi, _ = parent_a.indices_range['E']
    crossover_index = random.randrange(lo, hi)

    child = parent_a.clone()
    child.genes[:crossover_index] = parent_b.genes[:crossover_index].copy()
    return [child]

# 

# def n_point_crossover(crossover_rate: float, n_cross_T:int=1, n_cross_E:int=2 ):

#     def crossover(parents:List[Chromosome], gabw: GaHMM):
        
#         no_crossover = random.uniform(0,1) > crossover_rate
#         if no_crossover:
#             return parents

#         slice_points = numpy.empty(n_cross_T + n_cross_E + 3)
#         slice_points

#         lo, hi, _ = gabw.range['T']
#         slice_points[:n_cross_T] = npr.randint(lo, hi, size=n_cross_T)

#         lo, hi, _ = gabw.range['E']

#         slice_points[n_cross_T:n_cross_E] =  npr.randint(lo, hi, size=n_cross_E)

#         slice_points = [0, gabw.n_states, *sorted(slice_points), gabw.n_genes]

#         child_a_genes = []
#         child_b_genes = []
#         for i in range(len(slice_points) -1):
#             start = slice_points[i]
#             stop = slice_points[i+1]

#             child_a_genes += parents[i % 2].genes[start:stop]
#             child_b_genes += parents[1 - (i % 2)].genes[start:stop]

#         children = [
#             Chromosome(child_a_genes),
#             Chromosome(child_b_genes),
#         ]
#         return children


#     return crossover