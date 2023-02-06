import numpy
import random 
import ga.numba_ga as ga
from ga.types import ChromosomeSlices, ChromosomeMask, MutationFunction
from numba import jit
rng = numpy.random.default_rng()

def delete_random_emission_symbols(n_zeros: int):

    def mutation_func(chromosome: numpy.ndarray, slices: ChromosomeSlices, mask: ChromosomeMask, gabw: ga.GaHMM):
        chromosome = chromosome.copy()
        start, stop, _ = slices.emission_probs
        indices = numpy.random.randint(low=start, high=stop, size=n_zeros)
        for i in range(len(indices)):
            index = indices[i]
            chromosome[index] = 0

        return chromosome
    
    return mutation_func

def dynamic_uniform_mutation_factory(mutation_threshold: float, beta: float):

    def mutation_func(chromosome: numpy.ndarray, slices: ChromosomeSlices, mask: ChromosomeMask, gabw: ga.GaHMM):

        mutation_strength_mod = (1 - gabw.current_generation/gabw.n_generations)**beta

        chromosome = chromosome.copy()
        start, stop, _ = slices.emission_probs
        for i in range(start, stop):
            mutation_chance = random.uniform(0,1)
            if mutation_chance < mutation_threshold:
                direction = rng.choice((-1,1))
                mutatation_strength = rng.random()
                chromosome[i] += direction * (1 - mutatation_strength**mutation_strength_mod)
                        
        return chromosome
    
    return mutation_func

def constant_uniform_mutation_factory(mutation_threshold: float) -> MutationFunction:

    # @jit(nopython=True, cache=True, parallel=True)
    def mutation_func(chromosome: numpy.ndarray, gabw: ga.GaHMM):

        start, stop, _ = gabw.ranges.B
        for i in range(start, stop):
            mutation_chance = random.uniform(0,1)
            if mutation_chance < mutation_threshold:
                chromosome[i] = random.uniform(0,1) / gabw.n_symbols
                        
        return chromosome
    
    return mutation_func


def numba_constant_uniform_mutation(mutation_threshold: float):
    """_summary_

    Args:
        mutation_threshold (float): For every gene in the chromosome a random number is generated.
        If the random number is greater or equal than the mutation_threshold the gene is replaced by a random number
    """

    
    def mutation_func(children: numpy.ndarray, slices: ChromosomeSlices, mask: ChromosomeMask, gabw: ga.GaHMM):
        """chromosomes sind ne menge an children !! kein einzelnes chromosome

        Args:
            chromosome (numpy.ndarray): 2D numpy array
            slices (ga.ChromosomeSlices): _description_
            mask (numpy.ndarray): _description_
            gabw (ga.GaHMM): _description_

        Returns:
            _type_: _description_
        """

        chromosomes = children.copy()

        n_chromosomes, n_genes = chromosomes.shape
        for i in range(n_chromosomes):
            for j in range(n_genes):
                is_mutable = not mask[j]
                if is_mutable:
                    mutation_chance = random.uniform(0,1)
                    if mutation_chance <= mutation_threshold:
                        chromosomes[i, j] = random.uniform(0,1)
                

        return chromosomes
      
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

    
