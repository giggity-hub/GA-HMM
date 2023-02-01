import numpy

class FitnessBlock:
    def __init__(fitness_func) -> None:
        pass

    def execute(self, gabw):
        self.calculate_fitness(gabw)



    def calculate_fitness(self, gabw):
        for i in range(self.population_size):
            hmm_params = self.chromosome2hmm_params(self.population[i])
            log_prob = self.fitness_func(hmm_params)
            self.population[i, self.slices.fitness.start] = log_prob


    def chromosome2hmm_params(self, chromosome: numpy.ndarray):
        n_states = self.slices.transition_probs.step
        n_symbols = self.slices.emission_probs.step

        start, stop, _ = self.slices.start_probs
        start_vector = chromosome[start:stop]

        start, stop, _ = self.slices.emission_probs
        emission_matrix = chromosome[start:stop].reshape((n_states, n_symbols))

        start, stop, _ = self.slices.transition_probs
        transition_matrix = chromosome[start:stop].reshape((n_states, n_states))

        hmm_params = HmmParams(start_vector.copy(), emission_matrix.copy(), transition_matrix.copy())
        return hmm_params