import numpy
from pomegranate import HiddenMarkovModel, DiscreteDistribution
from hmmlearn.hmm import MultinomialHMM
from lib.utils import rand_stochastic_vector, rand_stochastic_matrix

def normalize_rows(prob_array):
    axis = prob_array.ndim -1
    row_sums = numpy.sum(prob_array, axis=axis, keepdims=True)
    # log(a/b) = log(a) - log(b)

    return prob_array / row_sums

def normalize_row(row):
    # row = row.round(2)
    row_sum = numpy.sum(row)
    normalized_row = row / row_sum 
    # normalized_row.round(2)
    # normalized_row_sum = normalize_row.sum()
    return normalized_row


# np.array((values - min(values)) / (max(values) - min(values)))


# def hmm_from_vector_hmmlearn(vector, n_states, alphabet):
#     n_symbols = len(alphabet)
#     start_probs = vector[:n_states]

#     trans_probs_vector = vector[n_states:(n_states*(n_states + 1))]
    
#     trans_probs_matrix = numpy.reshape(trans_probs_vector, (n_states, n_states))

#     state_probs_vector = vector[n_states*(n_states + 1):]
#     state_probs_matrix = numpy.reshape(state_probs_vector, (n_states, n_symbols))



#     model = MultinomialHMM(n_components=n_states, init_params='', n_trials=1)
#     model.n_features = n_symbols
#     model.emissionprob_ = state_probs_matrix 
#     model.startprob_ = start_probs
#     model.transmat_ = trans_probs_matrix
#     return model

def hmm_from_vector(vector, n_states, alphabet):
    n_symbols = len(alphabet)
    start_probs = vector[:n_states]

    trans_probs_vector = vector[n_states:(n_states*(n_states + 1))]
    
    trans_probs_matrix = numpy.reshape(trans_probs_vector, (n_states, n_states))

    state_probs_vector = vector[n_states*(n_states + 1):]
    state_probs_matrix = numpy.reshape(state_probs_vector, (n_states, n_symbols))
    state_probs_dists = [DiscreteDistribution(dict(zip(alphabet, probs))) for probs in state_probs_matrix]


    model = HiddenMarkovModel.from_matrix(trans_probs_matrix, state_probs_dists, start_probs)
    return model

def random_chromosome(num_states, alphabet):

    num_symbols = len(alphabet)
    # Start Probabilities
    # start_probs = numpy.random.rand(num_states)
    # normalized_start_probs = normalize_rows(start_probs)

    # # Transition Probabilities
    # transition_probs = numpy.random.rand(num_states, num_states)
    # normalized_transition_probs = normalize_rows(transition_probs)

    # # Emission Probabilities
    # emission_probs = numpy.random.rand(num_states, num_symbols)
    # normalized_emission_probs = normalize_rows(emission_probs)
    start_probs = rand_stochastic_vector(num_states)
    transition_probs = rand_stochastic_matrix(num_states, num_states)
    emission_probs = rand_stochastic_matrix(num_states, num_symbols)

    chromosome = start_probs + transition_probs.flatten() + emission_probs.flatten()

    return chromosome


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



# if __name__ == "__main__":
#     chromosome = random_chromosome(6, list('abcdefghi'))
#     model = hmm_from_vector_hmmlearn(chromosome, 6, list('abcdefghi'))
#     x, y = model.sample(4)
#     print(x,y)