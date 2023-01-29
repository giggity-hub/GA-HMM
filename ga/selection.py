# from ga.ga import GaHMM, Chromosome
from typing import List, Tuple, Callable, Annotated
import numpy.random as npr
import numpy
import ga.numba_ga as ga
from ga.types import SelectionFunction

# def pairwise_fittest(
#     population: List[Chromosome], 
#     n_offspring: int, 
#     gabw: GaHMM
#     ) -> List[Tuple[Chromosome, Chromosome]]:
    
#     parent_pairs = []
#     i = 0

#     while len(parent_pairs) < n_offspring:
#         parent1 = population[i % len(population)]
#         parent2 = population[(i+1) % len(population)]

#         parent_pairs.append((parent1, parent2))
#         i+=1

#     return parent_pairs



def rank_selection(population_size:int) -> SelectionFunction:
    """_summary_

    Args:
        population_size (int): 

    Returns:
        SelectionFunction: _description_
    """

    def gauss_sum(n):
        return (n**2 + n)//2
    
    total_rank = gauss_sum(population_size)
    selection_probs = numpy.arange(population_size, 0, -1) / total_rank

    def selectOne():
        return npr.choice(population_size, p=selection_probs)

    def selection_func(
        population: numpy.ndarray,
        n_offspring: int,
        slices: ga.ChromosomeSlices,
        gabw: ga.GaHMM
        ) -> numpy.ndarray:

        n_genes = population.shape[1]
        n_parents = n_offspring*2
        parents = numpy.empty((n_parents, n_genes))

        for i in range(n_parents -1):
            parent_a_index = selectOne()
            parent_b_index = selectOne()

            parents[i] = population[parent_a_index, :].copy()
            parents[i+1] = population[parent_b_index, :].copy()

        return parents.copy()

    return selection_func



# def rank_selection(
#     population: numpy.ndarray,
#     n_offspring: int,
#     slices: ga.ChromosomeSlices,
#     gabw: ga.GaHMM
#     ) -> numpy.ndarray:

#     population_size, n_genes = population.shape
#     def gauss_sum(n):
#         return (n**2 + n)//2

#     # max_rank = gabw.population_size
#     # def inverse_rank(x):
#     #     return population_size - x.rank

#     total_rank = gauss_sum(population_size)
#     selection_probs = numpy.zeros(population_size)
    
#     for i in range(population_size):
#         rank = population[i, slices.rank.start]
#         inverse_rank = population_size - rank
#         selection_probs[i] = inverse_rank/total_rank
    
#     # selection_probs = [inverse_rank(x)/total_rank for x in population]

#     def selectOne():
#         return population[npr.choice(gabw.population_size, p=selection_probs)]

#     n_genes = population.shape[1]
#     parents = numpy.empty((n_offspring*2, n_genes))
#     for i in range(n_offspring):
#         parent_a = selectOne()
#         parent_b = selectOne()

#         parents[i] = parent_a.copy()
#         parents[i+1] = parent_b.copy()

#     return parents
    

# def roulette_wheel_selection(
#     population: List[Chromosome],
#     n_offspring: int,
#     ga_instance: GaHMM
#     ) -> List[List[Chromosome]]:
    

#     selection_probs = [i.fitness for i in population]

#     def selectOne():
#         return population[npr.choice(ga_instance.population_size, p=selection_probs)]

#     parents = []

#     for i in range(n_offspring//2):
#         parent_a = selectOne()
#         parent_b = selectOne()
#         parents.append([parent_a, parent_b])

#     return parents