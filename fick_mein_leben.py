import pygad
import numpy


N_STATES = 6
ALPHABET = list('ABCDEFG')


def fitness_func(solution, solution_idx):
    return numpy.random.rand()



ga_instance = pygad.GA(num_generations=10,
                       sol_per_pop=5,
                    #    initial_population=initial_population,
                       num_parents_mating=2,
                       num_genes=20,
                       fitness_func=fitness_func,
                       crossover_type = "single_point",
                       keep_elitism=1,
                    #    on_stop=on_stop,
                       init_range_high=4,
                       init_range_low=1)

ga_instance.run()
ga_instance.plot_fitness()
