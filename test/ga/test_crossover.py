import pytest
from ga.types import CrossoverFunction
from ga.crossover import numba_single_point_crossover2
from ga.numba_ga import GaHMM
import numpy
import math


crossover_functions = [
    numba_single_point_crossover2
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

def test_same_length_as_parents(parents, child):
    n_genes = parents.shape[1]
    assert child.shape == (n_genes, )

# def test_child_differs_from_parents(parents, child):


def test_no_nan_values(child):
    assert not numpy.any(numpy.isnan(child))

def test_all_values_leq_one(child):
    assert numpy.all(numpy.less_equal(child, 1))

def test_all_values_geq_zero(child):
    assert numpy.all(numpy.greater_equal(child, 0))

def test_child_has_None_base(child):
    # If the Child would Share Memory with a parent the base would be that parent
    # We don't won't the child to share Memory with a parent therefore it's base should be None
    assert not type(child.base) == numpy.ndarray
    assert child.base == None

def test_parents_have_none_base(parents, child):
    assert not type(parents.base) == numpy.ndarray
    assert child.base == None