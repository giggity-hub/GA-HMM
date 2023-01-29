import pytest
from ga.types import CrossoverFunction
from ga.crossover import numba_single_point_crossover2, combine_parent_states
from ga.numba_ga import GaHMM
import numpy
import math
from test.assertions import assert_all_values_are_probabilities, assert_no_shared_memory

crossover_functions = [
    numba_single_point_crossover2,
    combine_parent_states
]

@pytest.fixture(params=crossover_functions)
def crossover_func(request):
    return request.param

@pytest.fixture
def gabw_mock():
    return GaHMM(
        n_symbols=128,
        n_states=4,
        population_size=13,
        n_generations=10
    )

@pytest.fixture
def parents(gabw_mock: GaHMM):
    n_parents = 2
    parents = numpy.random.rand(n_parents, gabw_mock.n_genes)
    return parents

@pytest.fixture
def child(parents, crossover_func: CrossoverFunction, gabw_mock: GaHMM):
    child = crossover_func(parents, gabw_mock.slices, gabw_mock)
    return child

def test_child_has_same_length_as_parents(parents, child):
    n_genes = parents.shape[1]
    assert child.shape == (n_genes, )


def test_all_child_values_are_probabilities(child):
    assert_all_values_are_probabilities(child)

def test_no_shared_memory(parents, child):
    assert_no_shared_memory(parents)
    assert_no_shared_memory(child)