import numpy
from typing import List, Callable, Annotated
# from ga.ga import Chromosome, GaHMM
import random
import ga.numba_ga as ga
from ga.types import ChromosomeSlices, CrossoverFunction
rng = numpy.random.default_rng()
from numba import njit, jit

@jit(nopython=True, cache=True, parallel=True)
def _crossover(parents, parent_indices, crossover_indices):
    n_children, n_crossover_points = parent_indices.shape
    n_parents, n_genes = parents.shape

    children = numpy.empty((n_children, n_genes))
    
    for child_index in range(n_children):
        children[child_index] = parents[child_index % n_parents].copy()
        for crossover_index in range(n_crossover_points):

            start = crossover_indices[crossover_index]
            stop = crossover_indices[crossover_index + 1]
            parent_index = parent_indices[child_index, crossover_index]

            children[child_index, start:stop] = parents[parent_index, start:stop].copy()
    
    return children


def calculate_rank_weighted_selection_probs(ranks: numpy.ndarray, population_size: int):
    """Calculate probabilities of being selected as parent proportional to parents rank.

    Args:
        parents (numpy.ndarray): _description_
        slices (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    inverse_ranks = population_size - ranks

    selection_probs = inverse_ranks / inverse_ranks.sum()
    return selection_probs

def rank_weighted(crossover_func: CrossoverFunction) -> CrossoverFunction:

    def weighted_crossover_func(parents: numpy.ndarray, n_children: int, slices: ChromosomeSlices, gabw: ga.GaHMM) -> numpy.ndarray:
        ranks = parents[:, slices.rank.start]
        selection_probs = calculate_rank_weighted_selection_probs(ranks, population_size=gabw.population_size)
        child = crossover_func(parents, n_children, slices, gabw, selection_probs)
        return child

    return weighted_crossover_func

def select_parent_indices(n_parents: int, n_children: int, n_crossover_points: int, selection_probs=None):
    """Returns parent indices array so that parent_indices[i, j] = which parent to choose for child i, gene j

    Args:
        n_parents (_type_): _description_
        n_children (_type_): _description_
        n_crossover_points (_type_): _description_
        selection_probs (_type_): _description_

    Returns:
        _type_: (n_children x n_crossover_points)
    """
    parent_indices = numpy.random.choice(n_parents, size=(n_crossover_points*n_children), p=selection_probs)
    parent_indices = parent_indices.reshape((n_children, n_crossover_points))
    return parent_indices


# def every_parent_at_least_once(parent_indices, n_parents):

#     # Will throw an error if len(parent_indices) < n_parents
#     rand_indices = rng.choice(len(parent_indices), size=n_parents, replace=False)

#     for parent_index in range(n_parents):
#         index = rand_indices[parent_index]
#         parent_indices[index] = parent_index
    
#     return parent_indices



def arithmetic_mean_crossover(parents: numpy.ndarray, unused_n_children: int, slices: ChromosomeSlices, gabw: ga.GaHMM, selection_probs=1) -> numpy.ndarray:
    # n_children has no effect since a group of parents only has one mean value
    child = parents[0].copy()
    n_parents = parents.shape[0]

    start, stop, _ = slices.emission_probs

    parents_weighted = parents[:, start:stop] * numpy.atleast_2d(selection_probs).T
    parents_sum = numpy.sum(parents_weighted, axis=0)
    parents_mean = parents_sum / n_parents

    child[start:stop] = parents_mean

    return numpy.atleast_2d(child)


# Can have infinite children (some may be same tho)
def uniform_crossover(parents: numpy.ndarray, n_children: int, slices: ChromosomeSlices, gabw: ga.GaHMM, selection_probs=None) -> numpy.ndarray:
    n_parents = parents.shape[0]
    low, high, _ = slices.emission_probs
    n_crossover_points = high - low
    
    crossover_indices = numpy.arange(low, high+1, step=1)
    parent_indices = select_parent_indices(n_parents, n_children, n_crossover_points, selection_probs)

    children = _crossover(parents, parent_indices, crossover_indices)
    return children

# Can not be weighted
# def n_point_crossover_factory(n_crossover_points: int, n_parents_per_mating: int, n_children_per_mating: int) -> CrossoverFunction:
#     """Supports usage of more than two parents per child. n_crossover_points should be at least one greater than the number of parents

#     Args:
#         n_crossover_points (int): _description_

#     Returns:
#         CrossoverFunction: _description_
#     """

#     expected_state_counts

#     if n_parents_per_mating != n_children_per_mating or (n_crossover_points + 1) < n_children_per_mating:

#     parents 5
#     children 3
#     crossoverpoints = 10
#     n_children = n_parents

#     if n_crossover_points < n_parents_per_mating


#     def crossover_func(parents: numpy.ndarray, n_children: int, slices: ChromosomeSlices, gabw: ga.GaHMM) -> numpy.ndarray:
#         low, high, _ = slices.emission_probs

#         n_parents_per_mating = parents.shape[0]
        
#         crossover_indices = numpy.empty(n_crossover_points + 2, dtype=int)
#         crossover_indices[0] = low
#         crossover_indices[1] = high
#         crossover_indices[2:] = numpy.random.randint(low, high, size=n_crossover_points)
#         crossover_indices.sort()
        

#         parent_indices_for_first_child = numpy.arange(n_crossover_points + 1)
#         parent_indices_repeated_across_second_dim = parent_indices_for_first_child * numpy.ones((1, n_children), dtype=int)
#         parent_indices_offset = parent_indices_repeated_across_second_dim + numpy.arange(n_children).reshape((n_children, 1))
#         parent_indices = parent_indices_offset % n_parents_per_mating

#         children = _crossover(parents, parent_indices, crossover_indices)
#         return children
    
#     return crossover_func

# def numba_single_point_crossover2(parents: numpy.ndarray, slices: ChromosomeSlices, gabw: ga.GaHMM) -> numpy.ndarray:
#     lo, hi, _ = slices.emission_probs
#     crossover_index = random.randrange(lo, hi)


#     child = numpy.zeros(parents.shape[1])
#     child[:crossover_index] = parents[0, :crossover_index].copy()
#     child[crossover_index:] = parents[1, crossover_index:].copy()
    
#     child = child.copy()
#     # child = parents[0, :].copy()
#     return child




def uniform_states_crossover(parents: numpy.ndarray, n_children: int, slices: ChromosomeSlices, gabw: ga.GaHMM, selection_probs=None) -> numpy.ndarray:

    parent_indices = select_parent_indices(
        n_parents= parents.shape[0], 
        n_children = n_children, 
        n_crossover_points= gabw.n_states,
        selection_probs=selection_probs)


    crossover_indices = numpy.empty(gabw.n_states*2 + 1, dtype=int)

    start, stop, step = slices.emission_probs
    crossover_indices[:gabw.n_states] = numpy.arange(start=start, stop=stop, step=step)

    start, stop, step = slices.transition_probs
    crossover_indices[gabw.n_states:] = numpy.arange(start=start, stop=stop+1, step=step)

 
    children = _crossover(parents, parent_indices, crossover_indices)
    return children

# weighted(arithmetic_mean_crossover)



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