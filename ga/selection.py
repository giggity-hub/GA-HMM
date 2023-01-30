# from ga.ga import GaHMM, Chromosome
from typing import List, Tuple, Callable, Annotated
import numpy.random as npr
import numpy
import ga.numba_ga as ga
from ga.types import SelectionFunction
from numba import jit

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





@jit(nopython=True, cache=True)
def _selection(population, n_parents, parent_indices):
    n_genes = population.shape[1]
    parents = numpy.empty((n_parents, n_genes))

    for i in range(n_parents):
        parent_index = parent_indices[i]
        parents[i] = population[parent_index, :].copy()

    return parents.copy()

def invert_and_normalize_scale(ndarr, ndarr_min, ndarr_max):
    ndarr_inverted =  -ndarr + ndarr_min + ndarr_max
    ndarr_normalized = ndarr_inverted / ndarr_inverted.sum()
    return ndarr_normalized


def stochastic_universal_sampling_selection(
    population: numpy.ndarray,
    n_parents: int,
    slices: ga.ChromosomeSlices,
    gabw: ga.GaHMM
    ) -> numpy.ndarray:

    """_summary_https://sci-hub.hkvisa.net/10.1007/978-3-540-73190-0_3
    """

    # create range with steps 1/n_parents
    # move that range by some random offset in range(0,1)
    # take modulo damit noch alle elemente wider im moped sind

    pointers = numpy.arange(0, 1, step=1/n_parents)
    pointers_offset = numpy.random.rand()
    pointers = (pointers + pointers_offset) % 1


def tournament_selection_factory(tournament_size):

    def selection_func(
        population: numpy.ndarray,
        n_parents: int,
        slices: ga.ChromosomeSlices,
        gabw: ga.GaHMM
        ) -> numpy.ndarray:

        tournament = npr.choice(len(population), size=n_parents*tournament_size).reshape((tournament_size, n_parents))
        tournament = numpy.sort(tournament, axis=0)

        tournament_selection_probs_scale = numpy.arange(tournament_size, 0, -1)
        tournament_selection_pros = tournament_selection_probs_scale / numpy.sum(tournament_selection_probs_scale)
        tournament_winners = npr.choice(tournament_size, size=n_parents, p=tournament_selection_pros)

        parents_indices = numpy.take(tournament, tournament_winners, axis=0)
        parents = numpy.take(parents, parents_indices)

        return parents

    return selection_func

    

def random_selection(
    population: numpy.ndarray,
    n_parents: int,
    slices: ga.ChromosomeSlices,
    gabw: ga.GaHMM
    ) -> numpy.ndarray:

    parent_indices = npr.choice(len(population), size=n_parents)
    parents = _selection(population, n_parents, parent_indices)
    return parents

def roulette_wheel_selection(
    population: numpy.ndarray,
    n_parents: int,
    slices: ga.ChromosomeSlices,
    gabw: ga.GaHMM
    ) -> numpy.ndarray:

    fitness_values = population[gabw.slices.fitness.start]
    
    selection_probs = invert_and_normalize_scale(fitness_values, fitness_values[0], fitness_values[-1])
    population_size = population.shape[0]

    parent_indices = npr.choice(population_size, p=selection_probs, size=n_parents)
    parents = _selection(population, n_parents, parent_indices)
    return parents



def rank_selection_factory(population_size:int) -> SelectionFunction:

    def gauss_sum(n):
        return (n**2 + n)//2
    
    total_rank = gauss_sum(population_size)
    selection_probs = numpy.arange(population_size, 0, -1) / total_rank
        

    def selection_func(
        population: numpy.ndarray,
        n_parents: int,
        slices: ga.ChromosomeSlices,
        gabw: ga.GaHMM
        ) -> numpy.ndarray:

        parent_indices = npr.choice(population_size, p=selection_probs, size=n_parents)
        parents = _selection(population, n_parents, parent_indices)
        return parents

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