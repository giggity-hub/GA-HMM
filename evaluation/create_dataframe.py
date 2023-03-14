# What we wanna do 
from ga.numba_ga import GaHMM
import itertools
from data.data import Dataset
import evaluation.params as params
import pandas


# def loopedy_loop(
#         n_states, 
#         n_symbols, 
#         crossover_func, 
#         mutation_func, 
#         observation_sequences, 
#         population_size, 
#         n_generations):
    
    
#     gabw = GaHMM(
#         n_symbols=n_symbols,
#         n_states=n_states,
#         population_size=population_size,
#         n_generations=n_generations,
#         observations=observation_sequences)
    
#     gabw.crossover_func = crossover_func
#     gabw.mutation_func = mutation_func

#     gabw.apply_bw_every_nth_gen = apply_bw_every_nth_generation
#     gabw.n_bw_iterations_per_gen = n_bw_iterations_per_gen
#     gabw.n_bw_iterations_after_ga = n_bw_iterations_after_ga
#     gabw.n_bw_iterations_before_ga = n_bw_iterations_before_ga


#     gabw.start()
#     # fittest_individual = 
#     # processing times
#     row = {
#         'dataset': dataset_name,
#         'population_size': population_size,
#         'n_generations': n_generations,
#         'mutation_func': mutation_func_name,
#         'crossover_func': crossover_func_name,
#         'fitness_before_mutation': population_fitness_before_mutation[pop_index],
#         'fitness_after_mutation': population_fitness_after_mutation[pop_index],
#         # 'n_seqs': n_seqs,
#         # 'n_symbols': n_symbols,
#         # 'n_states': n_states,
#     }

#     # Dataset Configuration
#     # n_states, n_symbols
#     # n_sequences
#     # dataset_name

dataset_configurations = {
    'fsdd_128M_4N_10S': {
        'dataset_name': 'fsdd',
        'category': 0,
        'n_symbols': 128,
        'n_states': 4,
        'n_sequences': 10,
    }
}

ga_strategies = {
    'strat_1': {
        'n_bw_iterations_before_ga': 0,
        'n_bw_iterations_after_ga': 0,
        'n_bw_iterations_per_gen': 0,
        'apply_bw_every_nth_gen': 1,
        'crossover_func_name': '1X',
        'mutation_func_name': 'UNIFORM_0.1',
        'selection_func_name': 'RANK_SELECTION',
        'n_generations': 20,
        'population_size': 30,
    }
}

n_runs_per_parameter_product = 5
indices = [i for i in range(n_runs_per_parameter_product)]
parameter_product = itertools.product(dataset_configurations.keys(), ga_strategies.keys(), indices)
# parameter_product = itertools.repeat(parameter_product, n_runs_per_parameter_product)
dataframe_rows = []

for (dataset_conf_key, ga_strat_key, i) in parameter_product:
    ga_strat = ga_strategies[ga_strat_key]
    dataset_conf = dataset_configurations[dataset_conf_key]

    dataset = Dataset(dataset_conf['dataset_name'], dataset_conf['n_symbols'])
    observations = dataset.get_first_n_observations_of_category(dataset_conf['category'], dataset_conf['n_sequences'])

    gahmm = GaHMM(
        n_symbols = dataset_conf['n_symbols'],
        n_states = dataset_conf['n_states'],
        population_size = ga_strat['population_size'],
        n_generations = ga_strat['n_generations'],
        observations = observations
    )

    gahmm.crossover_func = params.crossover_functions[ga_strat['crossover_func_name']]
    gahmm.mutation_func = params.mutation_functions[ga_strat['mutation_func_name']]
    gahmm.parent_select_func = params.selection_functions[ga_strat['selection_func_name']]
    gahmm.apply_bw_every_nth_gen = ga_strat['apply_bw_every_nth_gen']
    gahmm.n_bw_iterations_after_ga = ga_strat['n_bw_iterations_after_ga']
    gahmm.n_bw_iterations_before_ga = ga_strat['n_bw_iterations_before_ga']
    gahmm.n_bw_iterations_per_gen = ga_strat['n_bw_iterations_per_gen']


    _, fitness = gahmm.start()
    
    row = {
        'dataset_conf': dataset_conf_key,
        'ga_strat': ga_strat_key,
        'log_prob': fitness,
        'total_time': gahmm.performance_timers['total'].total_time,
        'crossover_time': gahmm.performance_timers['crossover'].total_time,
        'mutation_time': gahmm.performance_timers['mutation'].total_time,
        'fitness_time': gahmm.performance_timers['fitness'].total_time,
        'selection_time': gahmm.performance_timers['selection'].total_time
    }
    dataframe_rows.append(row)
    print(fitness)

crossover_df = pandas.DataFrame(dataframe_rows)
crossover_df.to_csv('evaluation/dataframe.csv')