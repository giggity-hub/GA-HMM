

import unittest
from hmm.bw import HMM
from lib.utils import rand_stochastic_matrix, rand_stochastic_vector
import numpy.random as npr

class Test(unittest.TestCase):
    def test_calc_alpha(self):

        n_states = 3
        n_symbols = 5

        pi = rand_stochastic_vector(n_states)
        b = rand_stochastic_matrix(n_states, n_symbols)
        a = rand_stochastic_matrix(n_states, n_states)

        T = 12
        O = npr.randint(low=0, high=n_symbols,  size=T)
        hmm = HMM(pi, b, a)
        prob = hmm.calc_alpha(O)
        print(prob)
    

if __name__ == '__main__':
    unittest.main()