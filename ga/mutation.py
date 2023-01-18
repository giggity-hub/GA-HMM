from lib.utils import rand_stochastic_vector, normalize_vector
import numpy
import random 
from ga.ga import Chromosome, GaHMM

def constant_uniform_mutation(mutation_threshold: float):
    """_summary_

    Args:
        mutation_threshold (float): For every gene in the chromosome a random number is generated.
        If the random number is greater or equal than the mutation_threshold the gene is replaced by a random number
    """

    
    def mutation_func(chromosome: Chromosome, gabw: GaHMM):
        mutation_chance = random.uniform(0,1)
        chromosome.range['E']
        start_index = gabw.range['E'][0]
        for i in range(gabw.len['ET']):
            if mutation_chance[i] >= mutation_threshold:
                chromosome.genes[start_index+i] = random.uniform(0,1)

        return chromosome
      
    return mutation_func


# def n_point_mutation(mutation_rate: float, n_mut_T:int=1, n_mut_E:int=2):

#     def mutation_func(chromosome: Chromosome, gabw: GaHMM):
#         no_mutation = random.uniform(0,1) > mutation_rate
#         if no_mutation:
#             return chromosome

#         mutation_indices = []
        
#         lo, hi, _ = gabw.range['T']
#         # mutation_indices += numpy.random.randint(lo, hi, size=n_mut_T)
#         mutation_indices += random.sample(range(lo,hi), k=n_mut_T)

#         lo, hi, _ = gabw.range['E']
#         # mutation_indices += numpy.random.randint(lo, hi, size=n_mut_E)
#         mutation_indices += random.sample(range(lo,hi), k=n_mut_E)
        
#         for i in mutation_indices:
#             chromosome.genes[i] = numpy.random.uniform(0,1)
        
#         return chromosome
        
#     return mutation_func





# # mutate one gene in every individual
# def mutate_one_gene(population, ga_instance={}):
#     for individual in population:
#         random_gene_index = numpy.random.choice(range(len(individual.genes)))
#         random_gene_length = len(individual.genes[random_gene_index])
#         individual.genes[random_gene_index] = rand_stochastic_vector(random_gene_length)

#     return population

# def mutate_one_bit(population, ga_instance={}):
#     for individual in population:
#         random_gene_index = numpy.random.choice(range(len(individual.genes)))
#         random_gene_length = len(individual.genes[random_gene_index])
#         random_bit_index = numpy.random.choice(range(random_gene_length))
#         individual.genes[random_gene_index][random_bit_index] = numpy.random.rand()
#         individual.genes[random_gene_index] = normalize_vector(individual.genes[random_gene_index])
    
#     return population
        
# def shuffle_one_gene(population, ga_instance={}):
#     for individual in population:
#         random_gene_index = numpy.random.choice(range(len(individual.genes)))
#         individual.genes[random_gene_index] = numpy.random.shuffle(individual.genes[random_gene_index])

#     return population

# def invert_one_gene(population, ga_instance={}):
#     for individual in population:
#         random_gene_index = numpy.random.choice(range(len(individual.genes)))
#         random_gene = individual.genes[random_gene_index]
#         # invert the probabilities
#         # muss noch normalisiert
#         individual.genes[random_gene_index] = normalize_vector([(1-i) for i in random_gene])
#     return population
# # mutates the individuals with a 
# # def mutate_below_avg_fitness(population, ga_instance):

# def switch_two_states(population, ga_instance={}):
#     for individual in population:
#         si_1, si_2 = random.sample(range(individual.n_states), 2)
#         tmp = individual.get_state_emissions(si_1).copy()
#         individual.set_state_emissions(si_1, individual.get_state_emissions(si_2).copy())
#         individual.set_state_emissions(si_2, tmp)

#     return population

    
