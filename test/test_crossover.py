from ga.crossover import numba_single_point_crossover




def test_numba_single_point_crossover(gabw):
    n_parents = 8
    parents = gabw.population[:n_parents, :]
    children = numba_single_point_crossover(parents, gabw.slices, gabw)

    n_genes = gabw.population.shape[1]
    assert children.shape == (n_parents//2, n_genes)
    # parent_a = digit_hmm_param_generator.next()
    # parent_b = digit_hmm_param_generator.next()
    # numba_single_point_crossover(parents=[parent_a, parek])
