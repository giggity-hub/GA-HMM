import numpy
from typing import List, Callable, Annotated
# from ga.ga import Chromosome, GaHMM
import random
import ga.numba_ga as ga
from ga.types import ChromosomeSlices


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

def numba_single_point_crossover2(parents: numpy.ndarray, slices: ChromosomeSlices, gabw: ga.GaHMM) -> numpy.ndarray:
    lo, hi, _ = slices.emission_probs
    crossover_index = random.randrange(lo, hi)


    child = numpy.zeros(parents.shape[1])
    child[:crossover_index] = parents[0, :crossover_index].copy()
    child[crossover_index:] = parents[1, crossover_index:].copy()
    
    child = child.copy()
    # child = parents[0, :].copy()
    return child

def combine_parent_states(parents: numpy.ndarray, slices: ChromosomeSlices, gabw: ga.GaHMM) -> numpy.ndarray:

    # 0 and 2**n -1 are excluded because the child should be never be the same as either parent
    one_below_only_ones_in_binary = 2**gabw.n_states -2
    crossover_pattern = numpy.random.randint(low=1, high=one_below_only_ones_in_binary)

    child = parents[0].copy()

    for state_index in range(gabw.n_states):
        parent_index = crossover_pattern & 1

        start, stop, _ = gabw.get_emission_probs_slice_for_state(state_index)
        child[start:stop] = parents[parent_index][start:stop].copy()

        start, stop, _ = gabw.get_transition_probs_slice_for_state(state_index)
        child[start:stop] = parents[parent_index][start:stop].copy()

        crossover_pattern = crossover_pattern >> 1

    return child
        


# def numba_single_point_crossover(parents: numpy.ndarray, slices: ChromosomeSlices ,gabw: ga.GaHMM=None) -> numpy.ndarray:
#     n_parents, n_genes = parents.shape
#     n_children = n_parents // 2
#     children = numpy.zeros((n_children, n_genes))
#     child_index = 0

#     for i in range(0, n_parents, 2):
#         lo, hi, _ = slices.emission_probs
#         crossover_index = random.randrange(lo, hi)

#         children[child_index, :crossover_index] = parents[i, :crossover_index].copy()
#         children[child_index, crossover_index:] = parents[i+1, crossover_index:].copy()

#     # child_genes = numpy.zeros((1, n_genes))
#     # child_genes[0, :crossover_index] = parents[0, :crossover_index].copy()
#     # child_genes[0, crossover_index:] = parents[1, crossover_index:].copy()

#     return children

    

# def single_point_crossover(parents: List[Chromosome], gabw: GaHMM=None):

#     parent_a, parent_b = parents
    
#     lo, hi, _ = parent_a.indices_range['E']
#     crossover_index = random.randrange(lo, hi)

#     child = parent_a.clone()
#     child.genes[:crossover_index] = parent_b.genes[:crossover_index].copy()
#     return [child]

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