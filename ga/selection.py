from ga.ga import GaHMM, Chromosome
from typing import List, Tuple
import numpy.random as npr

def pairwise_fittest(
    population: List[Chromosome], 
    n_offspring: int, 
    gabw: GaHMM
    ) -> List[Tuple[Chromosome, Chromosome]]:
    
    parent_pairs = []
    i = 0

    while len(parent_pairs) < n_offspring:
        parent1 = population[i % len(population)]
        parent2 = population[(i+1) % len(population)]

        parent_pairs.append((parent1, parent2))
        i+=1

    return parent_pairs

def rank_selection(
    population: List[Chromosome],
    n_offspring: int,
    gabw: GaHMM
    ) -> List[List[Chromosome]]:

    def gauss_sum(n):
        return (n**2 + n)//2

    # max_rank = gabw.population_size
    def inverse_rank(x):
        return gabw.population_size - x.rank

    total_rank = gauss_sum(gabw.population_size)
    selection_probs = [inverse_rank(x)/total_rank for x in population]

    def selectOne():
        return population[npr.choice(gabw.population_size, p=selection_probs)]

    parents = [(selectOne(), selectOne()) for i in range(n_offspring)]
    return parents
    

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