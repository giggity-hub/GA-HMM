# fitness function 1
# fitness with bound fixed length of input
from ga.ga import Individual, GeneticAlgorithm
from typing import List



def fixed_sample_fitness_func(samples: List[List[List[int]]]) -> callable[[Individual, GeneticAlgorithm], float]:
    """_summary_

    Args:
        samples (_type_): data should be a of size MxN where M is the number of classes (e.g 10 for Digit classification) 
        and N is the number of Observations for every class. An observation is an integer list.

    Returns:
        callable[[Individual, GeneticAlgorithm], float]: _description_
    """
    
    def fitness_func(individual: Individual, ga_instance: GeneticAlgorithm):
        pass
        # for every Zahl
        # get the fitness of classifier[zahl]
    
    # !!!!!!!!!!!!!!!!!! Die Parameter vom Child HMM sind nicht alle in range [0,1]
    # manche negativ manche größer als 1
    # vielleicht hab ich irgendwo reingespastet und die random generation ist nicht in der range [0,1]
    # child_hmm = hmm_from_vector(solution, N_STATES, ALPHABET)

    # total_score = []
    # for sample in samples:
    #     total_score.append(child_hmm.log_probability(sample))

    # mean_score = sum(total_score)/len(samples)
    # if mean_score > 0:
    #     x = child_hmm.forward_backward(sample)
    #     print('shish')
    # return mean_score
    return fitness_func