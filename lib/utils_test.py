from utils import rand_stochastic_matrix


def test_rand_stochastic_matrix():
    n = 4
    m = 128 
    print(rand_stochastic_matrix(n,m))


test_rand_stochastic_matrix()
