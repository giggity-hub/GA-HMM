import ga.crossover as crossover
import ga.mutation as mutation

crossover_functions = {
    '1X': crossover.n_point_crossover_factory(n_crossover_points=1),
    '2X': crossover.n_point_crossover_factory(n_crossover_points=2),
    'ARITHMETIC_MEAN': crossover.arithmetic_mean_crossover,
    'UNIFORM': crossover.uniform_crossover,
}

mutation_functions = {
    'UNIFORM_0.1': mutation.constant_uniform_mutation_factory(0.1),
    'UNIFORM_0.01': mutation.constant_uniform_mutation_factory(0.01),
    'UNIFORM_0.001': mutation.constant_uniform_mutation_factory(0.001),
}

tied = [
    {
        'dataset_name': 'fsdd',
        'n_symbols': 128,
        'n_seqs': 10,
        'observation_category': 0,
        'n_states': 5,
        'population_size': 100000,
    }
    # {
    #     'dataset_name': 'orl',
    #     'n_symbols': 16,
    #     'n_seqs': 10,
    #     'observation_category': 0,
    #     'n_states': 20,
    #     'population_size': 10,
    # }
]

max_n_bw_iterations = 20

