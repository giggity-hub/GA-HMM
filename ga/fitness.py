# fitness function 1
# fitness with bound fixed length of input
from ga.ga import Individual, GeneticAlgorithm
from typing import List, Callable


def log_prob_fitness(samples: List[List[int]]) -> Callable[[Individual, GeneticAlgorithm], float]:
    """_summary_

    Args:
        samples should be a list of observations

    Returns:
        callable[[Individual, GeneticAlgorithm], float]: _description_
    """
    

    def fitness_func(individual: Individual, ga_instance: GeneticAlgorithm=None) -> float:

        child_hmm = individual.to_hmm()

        total_score = 0
        for sample in samples:
            total_score += child_hmm.log_probability(sample)
        
        mean_score = total_score / len(samples)
        return mean_score
    

    return fitness_func