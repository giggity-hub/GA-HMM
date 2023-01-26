from ga.types import SelectionFunction
from ga.selection import rank_selection
import pytest
from ga.numba_ga import GaHMM
import numpy

selection_functions = [
    rank_selection(population_size=13)
]

@pytest.fixture(params=selection_functions)
def selection_func(request):
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
def n_offspring():
    return 10

@pytest.fixture
def parents(selection_func: SelectionFunction, n_offspring, gabw_mock: GaHMM):
    parents = selection_func(gabw_mock.population, n_offspring, gabw_mock.slices, gabw_mock)
    return parents


def test_parents_shape(parents, n_offspring, gabw_mock: GaHMM):
    assert parents.shape == (n_offspring*2 ,gabw_mock.n_genes)


def test_population_unchanged(selection_func, gabw_mock: GaHMM, n_offspring):
    population_before_selection = gabw_mock.population.copy()
    _ = selection_func(gabw_mock.population, n_offspring, gabw_mock.slices, gabw_mock)

    assert numpy.array_equal(population_before_selection, gabw_mock.population)





