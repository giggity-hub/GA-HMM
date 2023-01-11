from ga.ga import GeneticAlgorithm, Individual
from typing import List
import numpy.random as npr

def pairwise_fittest(
    population: List[Individual], 
    n_parent_pairs: int, 
    ga_instance: GeneticAlgorithm
    ) -> List[List[Individual]]:
    
    parent_pairs = []
    i = 0

    while len(parent_pairs) < n_parent_pairs:
        parent1 = population[i % len(population)]
        parent2 = population[(i+1) % len(population)]

        parent_pairs.append([parent1, parent2])
        i+=1

    return parent_pairs

def roulette_wheel_selection(
    population: List[Individual],
    n_parent_pairs: int,
    ga_instance: GeneticAlgorithm
    ) -> List[List[Individual]]:
    
    total_fitness = sum([i.fitness for i in population])
    # Die Selection probabilities müssen invertiert werden, da
    selection_probs = [(1 - i.fitness/total_fitness) for i in population]

    def selectOne():
        return population[npr.choice(len(population), p=selection_probs)]

    parents = []

    for i in range(n_parent_pairs/2):
        parent_a = selectOne()
        parent_b = selectOne()
        parents.append([parent_a, parent_b])

    return parents
