from ga.numba_ga import GaHMM
from data.data import Observations
import pandas
import numpy
import seaborn as sns
import hmm.bw as bw
import ga.representation as representation
from ga.numba_ga import normalize_population, calculate_fitness_of_population, train_population_with_bw
import evaluation.params as params

crossover_df_rows = []

for tied_params in params.tied:
    (dataset_name, n_symbols, n_seqs, observation_category, n_states, population_size) = tied_params.values()

    dataset = Observations(dataset_name, n_symbols)
    observation_seqs = dataset.get_first_n_observations_of_category(observation_category, n_seqs)
    
    gabw = GaHMM(
        n_symbols,
        n_states,
        population_size,
        n_generations=0,
        observations=observation_seqs)
    gabw.bake()

    parents_fitness = calculate_fitness_of_population(gabw.population, observation_seqs)

    for n_bw_iterations in range(params.max_n_bw_iterations + 1):

        for crossover_func_name, crossover_func in params.crossover_functions.items():
            # gabw.mutation_func = params.mutation_functions[mutation_func_name]
            gabw.crossover_func = crossover_func

            children = gabw.do_crossover_step(gabw.population.copy())
            normalized_children = normalize_population(children)

            children_fitness = calculate_fitness_of_population(normalized_children, observation_seqs)

            for child_index in range(len(children)):
                parent_1_index = child_index * 2
                parent_2_index = parent_1_index + 1

                row = {
                    'dataset': dataset_name,
                    'n_seqs': n_seqs,
                    'n_symbols': n_symbols,
                    'n_states': n_states,
                    'n_bw_iterations': n_bw_iterations,
                    'crossover_func': crossover_func_name,
                    'parent1_fitness': parents_fitness[parent_1_index],
                    'parent2_fitness': parents_fitness[parent_2_index],
                    'child_fitness': children_fitness[child_index],
                }
                crossover_df_rows.append(row)
        
        gabw.population = train_population_with_bw(gabw.population, observation_seqs)
        parents_fitness = calculate_fitness_of_population(gabw.population, observation_seqs)


crossover_df = pandas.DataFrame(crossover_df_rows)
crossover_df.to_csv('evaluation/crossover_df.csv')