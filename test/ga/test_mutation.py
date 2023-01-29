from ga.mutation import numba_constant_uniform_mutation2
from ga.types import MutationFunction
from ga.numba_ga import GaHMM
import numpy
import pytest

mutation_functions = [
    numba_constant_uniform_mutation2(mutation_threshold=0)
]

@pytest.fixture(params=mutation_functions)
def mutation_func(request):
    return request.param

@pytest.fixture
def gabw_mock():
    return GaHMM(
        n_symbols=128,
        n_states=4,
        population_size=13,
        n_generations=100
    )

@pytest.fixture
def chromosome_before_mutation(gabw_mock: GaHMM):
    chromosome = numpy.random.rand(gabw_mock.n_genes)
    return chromosome

@pytest.fixture
def chromosome_mask(gabw_mock):
    mask = numpy.random.randint(low=0, high=2, size=gabw_mock.n_genes, dtype=bool)
    return mask

@pytest.fixture
def chromosome_mask_full(gabw_mock):
    mask = numpy.ones(gabw_mock.n_genes, dtype=bool)
    return mask

@pytest.fixture
def chromosome_mask_empty(gabw_mock):
    mask = numpy.zeros(gabw_mock.n_genes, dtype=bool)
    return mask


@pytest.fixture
def chromosome_after_mutation(chromosome_before_mutation, mutation_func: MutationFunction, chromosome_mask, gabw_mock):
    chromosome =  mutation_func(chromosome_before_mutation, gabw_mock.slices, chromosome_mask, gabw_mock)
    return chromosome


def test_no_nan_values(chromosome_after_mutation):
    assert not numpy.any(numpy.isnan(chromosome_after_mutation))

def test_all_values_leq_one(chromosome_after_mutation):
    assert numpy.all(numpy.less_equal(chromosome_after_mutation, 1))

def test_all_values_geq_zero(chromosome_after_mutation):
    assert numpy.all(numpy.greater_equal(chromosome_after_mutation, 0))

def test_same_shape(chromosome_before_mutation, chromosome_after_mutation):
    assert chromosome_before_mutation.shape == chromosome_after_mutation.shape

def test_no_masked_values_changed(chromosome_before_mutation, chromosome_mask, chromosome_after_mutation):
    masked_before_mutation = chromosome_before_mutation * chromosome_mask
    masked_after_mutation = chromosome_after_mutation * chromosome_mask
    assert numpy.array_equal(masked_before_mutation, masked_after_mutation)


def test_no_values_changed_with_full_mask(chromosome_before_mutation, chromosome_mask_full, mutation_func: MutationFunction, gabw_mock):
    chromosome = mutation_func(chromosome_before_mutation, gabw_mock.slices, chromosome_mask_full, gabw_mock)

    assert numpy.array_equal(chromosome_before_mutation, chromosome)

# def test_all_values_changed_with_empty_mask(chromosome_before_mutation, chromosome_mask_empty, mutation_func: MutationFunction, gabw_mock):
#     chromosome = mutation_func(chromosome_before_mutation, gabw_mock.slices, chromosome_mask_empty, gabw_mock)

#     difference = chromosome_before_mutation - chromosome
#     assert numpy.all(difference !=0)



# def test_numba_constant_uniform_mutation(gabw: GaHMM):
#     mutation_func = numba_constant_uniform_mutation(mutation_threshold=0.1)

#     n_children = 6
#     children = gabw.population[:n_children, :]
#     mutated_children = mutation_func(children, gabw.slices, gabw.chromosome_mask, gabw)
    
#     assert mutated_children.shape == children.shape

#     masked_children = children * gabw.chromosome_mask
#     masked_mutated_children = mutated_children * gabw.chromosome_mask
#     assert numpy.array_equal(masked_children, masked_mutated_children)

#     # assert False
