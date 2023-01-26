from ga.selection import rank_selection
from ga.numba_ga import GaHMM


def test_rank_selection(gabw: GaHMM):
    n_offspring = 10
    parents = rank_selection(gabw.population, n_offspring, gabw.slices, gabw)
    n_genes = gabw.population.shape[1]
    assert parents.shape == (n_offspring * 2, n_genes)