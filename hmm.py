import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution

def normalize_rows(prob_array):
    axis = prob_array.ndim -1
    row_sums = numpy.sum(prob_array, axis=axis, keepdims=True)
    return prob_array / row_sums


class MatrixRepresentation:
    def __init__(self, num_states, alphabet, genes=[]) -> None:
        self.num_states = num_states
        self.num_symbols = len(alphabet)
        self.alphabet = alphabet



        # Start Probabilities
        start_vector = numpy.random.rand(self.num_states)
        self.start_probs = normalize_rows(start_vector)

        # Transition Probabilities
        transition_matrix = numpy.random.rand(self.num_states, self.num_states)
        self.transition_probs = normalize_rows(transition_matrix)

        # Emission Probabilities
        emission_matrix = numpy.random.rand(self.num_states, self.num_symbols)
        self.emission_probs = normalize_rows(emission_matrix)


    def toHMM(self):
        emission_distributions = [DiscreteDistribution(dict(zip(self.alphabet, probs))) for probs in self.emission_probs]
        model = HiddenMarkovModel(self.transition_probs, emission_distributions, self.start_probs)
        return model


class ChromosomeHMM:
    def __init__(self, num_states, alphabet, genes=[]) -> None:
        # define slicing points for the three parameters
        # create one big ass vector from the genes
        self.num_states = num_states 
        self.num_symbols = len(alphabet)
        self.alphabet = alphabet

        self.genes = genes if genes else self._init_genes()

    def start_genes(self):
        return self.genes[:self.num_states]

    def transition_genes(self):
        num_transitions = self.num_states**2
        return self.genes[self.num_states:num_transitions+self.num_states]

    def emission_genes(self):
        num_transitions = self.num_states**2
        return self.genes[num_transitions+self.num_states:]

    def _init_genes(self):
        # Start Probabilities
        start_probs = numpy.random.rand(self.num_states)
        normalized_start_probs = normalize_rows(start_probs)

        # Transition Probabilities
        transition_probs = numpy.random.rand(self.num_states, self.num_states)
        normalized_transition_probs = normalize_rows(transition_probs)

        # Emission Probabilities
        emission_probs = numpy.random.rand(self.num_states, self.num_symbols)
        normalized_emission_probs = normalize_rows(emission_probs)

        return [
            *normalized_start_probs,
            *normalized_transition_probs.flatten(),
            *normalized_emission_probs.flatten()
        ]

            
    def toHMM(self):
        start_probs = self.start_genes()

        state_probs_vector = self.emission_genes()
        state_probs_matrix = numpy.reshape(state_probs_vector, (self.num_states, self.num_symbols))
        state_probs_dists = [DiscreteDistribution(dict(zip(self.alphabet, probs))) for probs in state_probs_matrix]

        trans_probs_vector = self.transition_genes()
        trans_probs_matrix = numpy.reshape(trans_probs_vector, (self.num_states, self.num_states))

        model = HiddenMarkovModel.from_matrix(trans_probs_matrix, state_probs_dists, start_probs)
        return model

