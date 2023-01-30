import pytest
from ga.types import CrossoverFunction

from ga.numba_ga import GaHMM
import numpy
import math
from test.assertions import assert_all_values_are_probabilities, assert_no_shared_memory
from ga.crossover import ( 
    uniform_states_crossover,
    # n_point_crossover_factory,
    arithmetic_mean_crossover,
    uniform_crossover,
    rank_weighted)

crossover_functions = [
    uniform_states_crossover,
    rank_weighted(uniform_states_crossover),
    # n_point_crossover_factory(1),
    # n_point_crossover_factory(2),
    # n_point_crossover_factory(3),
    arithmetic_mean_crossover,
    rank_weighted(arithmetic_mean_crossover),
    uniform_crossover,
    rank_weighted(uniform_crossover)
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
    child = crossover_func(parents, gabw_mock.n_children_per_mating ,gabw_mock.slices, gabw_mock)
    return child

def test_children_shape(gabw_mock: GaHMM, child):
    assert child.shape == (gabw_mock.n_children_per_mating, gabw_mock.n_genes)


# def test_child_genes_are_masked(child, gabw_mock: GaHMM):


def test_all_child_values_are_probabilities(child):
    assert_all_values_are_probabilities(child)


def test_crossover_does_not_modify_parents(parents, crossover_func: CrossoverFunction, gabw_mock: GaHMM):
    parents_before_crossover = parents.copy()
    _ = crossover_func(parents, gabw_mock.n_children_per_mating ,gabw_mock.slices, gabw_mock)
    assert numpy.array_equal(parents_before_crossover, parents)

# def test_no_shared_memory(parents, child, crossover_func):
#     assert_no_shared_memory(parents)
#     assert_no_shared_memory(child)