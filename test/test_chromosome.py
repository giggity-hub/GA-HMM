import pytest
from hmm.hmm import random_left_right_hmm_params
from ga.ga import Chromosome
import numpy
from pytest import approx

def create_unnormalized_hmm_params(n_states, n_symbols):
    start_vector = numpy.random.rand(n_states)
    emission_matrix = numpy.random.rand(n_states, n_symbols)
    transition_matrix = numpy.random.rand(n_states, n_states)
    return (start_vector, emission_matrix, transition_matrix)

hmm_params = [
    random_left_right_hmm_params(n_states=5, n_symbols=128),
    random_left_right_hmm_params(n_states=2, n_symbols=10),
    random_left_right_hmm_params(n_states=3, n_symbols=3)
]

unnormalized_hmm_params = [
    create_unnormalized_hmm_params(n_states=5, n_symbols=128),
    create_unnormalized_hmm_params(n_states=2, n_symbols=10)
]


class Test:
    @pytest.mark.parametrize("start_vector, emission_matrix, transition_matrix", hmm_params)
    def test_constructor(self, start_vector, emission_matrix, transition_matrix):

        chromosome = Chromosome(start_vector, emission_matrix, transition_matrix)

        assert numpy.array_equal(start_vector, chromosome.start_vector)
        assert numpy.array_equal(transition_matrix, chromosome.transition_matrix)
        assert numpy.array_equal(emission_matrix, chromosome.emission_matrix)


    @pytest.mark.parametrize("start_vector, emission_matrix, transition_matrix", unnormalized_hmm_params)
    def test_normalize(self, start_vector, emission_matrix, transition_matrix):
        chromosome = Chromosome(start_vector, emission_matrix, transition_matrix)
        n_states = len(start_vector)

        chromosome.normalize()

        assert numpy.sum(chromosome.start_vector) == approx(1)
        assert numpy.min(chromosome.start_vector) >= 0
        assert numpy.max(chromosome.start_vector) <= 1

        assert numpy.sum(chromosome.emission_matrix, axis=1) == approx(numpy.ones(n_states))
        assert numpy.min(chromosome.emission_matrix) >= 0
        assert numpy.max(chromosome.emission_matrix) <= 1

        assert numpy.sum(chromosome.transition_matrix, axis=1) == approx(numpy.ones(n_states))
        assert numpy.min(chromosome.transition_matrix) >= 0 
        assert numpy.max(chromosome.transition_matrix) <= 1