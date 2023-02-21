import numpy
from hmm.types import MultipleHmmParams, HmmParams
from typing import List, NamedTuple, Tuple
import numpy.typing as npt
from ga.types import (
    Population, 
    Chromosome,
    ChromosomeFields,
    RangeTuple,
    ChromosomeRanges,
    ChromosomeSlices,
    ChromosomeMask
)


FIELDS = ChromosomeFields(
    N_STATES=  -1,
    N_SYMBOLS= -2,
    FITNESS=   -3,
    RANK=      -4
)


def calc_chromosome_length(n_states: int, n_symbols: int) -> int:
    n_genes = n_states * (1 + n_symbols + n_states)
    n_fields = len(FIELDS)
    chromosome_length = n_genes + n_fields
    return chromosome_length

def initialize_chromosome_fields(n_states, n_symbols, n_chromosomes):
    chromosome_fields = numpy.zeros((n_chromosomes, len(FIELDS)))
    chromosome_fields[:, FIELDS.N_STATES] = n_states
    chromosome_fields[:, FIELDS.N_SYMBOLS] = n_symbols
    chromosome_fields[:, FIELDS.FITNESS] = float('-inf')
    chromosome_fields[:, FIELDS.RANK] = numpy.arange(n_chromosomes)
    return chromosome_fields


def multiple_hmm_params_as_population(hmm_params: MultipleHmmParams) -> Population:
    PIs, Bs, As = hmm_params
    n_hmms, n_states, n_symbols = Bs.shape

    PI_genes = PIs
    B_genes = Bs.reshape((n_hmms, n_states * n_symbols))
    A_genes = As.reshape((n_hmms, n_states * n_states))

    chromosome_fields = initialize_chromosome_fields(n_states, n_symbols, n_hmms)

    chromosomes = numpy.hstack((
        PI_genes,
        B_genes,
        A_genes,
        chromosome_fields
    ))
    return chromosomes

def hmm_params_list_as_multiple_hmm_params(hmm_params_list: List[HmmParams]) -> MultipleHmmParams:
    PIs, Bs, As = map(numpy.array, zip(*hmm_params_list))
    return MultipleHmmParams(PIs, Bs, As)
    


def hmm_params_as_multiple_hmm_params(hmm_params: HmmParams) -> MultipleHmmParams:
    PIs, Bs, As = (numpy.expand_dims(param, axis=0) for param in hmm_params)
    return MultipleHmmParams(PIs, Bs, As)

def hmm_params_as_chromosome(hmm_params: HmmParams) -> Chromosome:
    multiple_hmm_params = hmm_params_as_multiple_hmm_params(hmm_params)
    population = multiple_hmm_params_as_population(multiple_hmm_params)
    chromosome = population[0]
    return chromosome

def calc_starts_and_stops(n_states: int, n_symbols: int) -> Tuple[numpy.array, numpy.array]:
    lengths = (n_states, (n_states*n_symbols), (n_states**2))
    stops = numpy.cumsum(lengths)
    starts = stops - lengths 
    return starts, stops



def calc_chromosome_slices(n_states: int, n_symbols: int) -> ChromosomeSlices:

    starts, stops = calc_starts_and_stops(n_states, n_symbols)

    PI_slice = slice(starts[0], stops[0])
    B_slice = slice(starts[1], stops[1])
    A_slice = slice(starts[2], stops[2])

    return ChromosomeSlices(PI_slice, B_slice, A_slice)



def calc_chromosome_ranges(n_states: int, n_symbols: int) -> ChromosomeRanges:
    n_states = int(n_states)
    n_symbols = int(n_symbols)
    
    starts, stops = calc_starts_and_stops(n_states, n_symbols)
    steps = [n_states, n_symbols, n_states]


    PI_range, B_range, A_range =  zip(starts, stops, steps)
    return ChromosomeRanges(
        RangeTuple(*PI_range), 
        RangeTuple(*B_range), 
        RangeTuple(*A_range))


def population_as_multiple_hmm_params(population: Population) -> MultipleHmmParams:
    n_hmms = len(population)
    n_states = int(population[0, FIELDS.N_STATES])
    n_symbols = int(population[0, FIELDS.N_SYMBOLS])

    PI_slice, B_slice, A_slice = calc_chromosome_slices(n_states, n_symbols)

    PIs = population[:, PI_slice]
    Bs = population[:, B_slice].reshape((n_hmms, n_states, n_symbols))
    As = population[:, A_slice].reshape((n_hmms, n_states, n_states))

    hmm_params = MultipleHmmParams(PIs, Bs, As)
    return hmm_params

def chromosome_as_population(chromosome):
    population = numpy.expand_dims(chromosome, axis=0)
    return population

def chromosome_as_hmm_params(chromosome):
    population = chromosome_as_population(chromosome)
    multiple_hmm_params = population_as_multiple_hmm_params(population)
    PI = multiple_hmm_params.PIs[0]
    B = multiple_hmm_params.Bs[0]
    A = multiple_hmm_params.As[0]

    hmm_params = HmmParams(PI, B, A)
    return hmm_params



def calculate_chromosome_mask(chromosome: Chromosome) -> ChromosomeMask:
    
    masked_genes = numpy.ma.masked_where((chromosome == 0) | (chromosome == 1), chromosome)
    mask = masked_genes.mask

    field_indices = numpy.array(FIELDS)
    mask[field_indices] = True

    return mask
