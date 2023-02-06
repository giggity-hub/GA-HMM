from hmm.types import MultipleHmmParams, HmmParams
from ga.types import Chromosome, Population

class Representation:

    

    def __init__(self) -> None:
        pass

    def calculate_slices(self, n_states: int, n_symbols: int) -> ChromosomeSlices:
        len_start_probs = n_states
        len_transition_probs = n_states*n_states
        len_emission_probs = n_states * n_symbols

        slice_start_probs = SliceTuple(0, len_start_probs, n_states)
        slice_emission_probs = SliceTuple(slice_start_probs.stop, slice_start_probs.stop +  len_emission_probs, n_symbols)
        slice_transition_probs = SliceTuple(slice_emission_probs.stop, slice_emission_probs.stop + len_transition_probs, n_states)

        slice_fitness = SliceTuple(slice_transition_probs.stop, slice_transition_probs.stop + 1, 1)
        slice_rank = SliceTuple(slice_fitness.stop, slice_fitness.stop + 1, 1)


        chromosome_slices = ChromosomeSlices(
            slice_start_probs,
            slice_emission_probs,
            slice_transition_probs,
            slice_fitness,
            slice_rank
        )

        return chromosome_slices
        

    def chromosome2hmm_params(self, chromosome: Chromosome):
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

    

    def hmm_params2chromosome(self, hmm_params: HmmParams):
        n_genes = self.slices.rank.stop
        chromosome=numpy.zeros(n_genes)

        start, stop, _ = self.slices.start_probs
        chromosome[start: stop] = hmm_params.start_vector
        start, stop, _ = self.slices.emission_probs
        chromosome[start: stop] = hmm_params.emission_matrix.flatten()
        start, stop, _ = self.slices.transition_probs
        chromosome[start: stop] = hmm_params.transition_matrix.flatten()

        return chromosome

    def hmms_as_population(self, all_hmm_params: MultipleHmmParams) -> Population:

        start_vectors, emission_matrices, transition_matrices = all_hmm_params

        start, stop, step = self.slices.start_probs
        self.population[:, start:stop] = start_vectors

        start, stop, step = self.slices.emission_probs
        self.population[:, start:stop] = emission_matrices.reshape((self.population_size, self.n_states * step))

        start, stop, step = self.slices.transition_probs
        self.population[:, start:stop] = transition_matrices.reshape((self.population_size, self.n_states * step))


    def population_as_hmms(self, population: Population) -> MultipleHmmParams: 
        start, stop, step = self.slices.start_probs
        start_vectors = self.population[:, start: stop]

        start, stop, step = self.slices.emission_probs
        emission_matrices = self.population[:, start: stop].reshape((self.population_size, self.n_states,step))

        start, stop, step = self.slices.transition_probs
        transition_matrices = self.population[:, start: stop].reshape((self.population_size, self.n_states, step))

        all_hmm_params = MultipleHmmParams(start_vectors, emission_matrices, transition_matrices)

        return all_hmm_params



        