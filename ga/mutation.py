from lib.utils import rand_stochastic_vector, normalize_vector
import numpy
import random 


# mutate one gene in every individual
def mutate_one_gene(population, ga_instance={}):
    for individual in population:
        random_gene_index = numpy.random.choice(range(len(individual.genes)))
        random_gene_length = len(individual.genes[random_gene_index])
        individual.genes[random_gene_index] = rand_stochastic_vector(random_gene_length)

    return population

def mutate_one_bit(population, ga_instance={}):
    for individual in population:
        random_gene_index = numpy.random.choice(range(len(individual.genes)))
        random_gene_length = len(individual.genes[random_gene_index])
        random_bit_index = numpy.random.choice(range(random_gene_length))
        individual.genes[random_gene_index][random_bit_index] = numpy.random.rand()
        individual.genes[random_gene_index] = normalize_vector(individual.genes[random_gene_index])
    
    return population
        
def shuffle_one_gene(population, ga_instance={}):
    for individual in population:
        random_gene_index = numpy.random.choice(range(len(individual.genes)))
        individual.genes[random_gene_index] = numpy.random.shuffle(individual.genes[random_gene_index])

    return population

def invert_one_gene(population, ga_instance={}):
    for individual in population:
        random_gene_index = numpy.random.choice(range(len(individual.genes)))
        random_gene = individual.genes[random_gene_index]
        # invert the probabilities
        # muss noch normalisiert
        individual.genes[random_gene_index] = normalize_vector([(1-i) for i in random_gene])
    return population
# mutates the individuals with a 
# def mutate_below_avg_fitness(population, ga_instance):

def switch_two_states(population, ga_instance={}):
    for individual in population:
        si_1, si_2 = random.sample(range(individual.n_states), 2)
        tmp = individual.get_state_emissions(si_1).copy()
        individual.set_state_emissions(si_1, individual.get_state_emissions(si_2).copy())
        individual.set_state_emissions(si_2, tmp)

    return population

    
