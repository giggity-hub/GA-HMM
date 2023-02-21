from ga.numba_ga import GaHMM
from data.data import Observations
import pandas
import numpy
import seaborn as sns
import hmm.bw as bw
import ga.representation as representation
from ga.numba_ga import normalize_population, calculate_fitness_of_population, train_population_with_bw
import evaluation.params as params

mutation_df_rows = []

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

    population_fitness_before_mutation = calculate_fitness_of_population(gabw.population, observation_seqs)

    for n_bw_iterations in range(params.max_n_bw_iterations + 1):

        for mutation_func_name, mutation_func in params.mutation_functions.items():
            gabw.mutation_func = params.mutation_functions[mutation_func_name]

            mutated_population = gabw.do_mutation_step(gabw.population.copy())
            normalized_population = normalize_population(mutated_population)
            population_fitness_after_mutation = calculate_fitness_of_population(normalized_population, observation_seqs)

            for pop_index in range(population_size):
                row = {
                    'dataset': dataset_name,
                    'n_seqs': n_seqs,
                    'n_symbols': n_symbols,
                    'n_states': n_states,
                    'n_bw_iterations': n_bw_iterations,
                    'mutation_func': mutation_func_name,
                    'fitness_before_mutation': population_fitness_before_mutation[pop_index],
                    'fitness_after_mutation': population_fitness_after_mutation[pop_index],
                }
                mutation_df_rows.append(row)
        
        gabw.population = train_population_with_bw(gabw.population, observation_seqs)
        population_fitness_before_mutation = calculate_fitness_of_population(gabw.population, observation_seqs)


mutation_df = pandas.DataFrame(mutation_df_rows)
mutation_df.to_csv('evaluation/mutation_df.csv')