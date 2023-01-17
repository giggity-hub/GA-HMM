import numpy
from typing import Tuple
from lib.utils import rand_stochastic_matrix, rand_stochastic_vector
import numpy.random as npr




class HMM:
    def __init__(self, pi: numpy.ndarray, b: numpy.ndarray, a: numpy.ndarray) -> None:
        self.pi = pi 
        self.a = a 
        self.b = b
        self.N = b.shape[0] # N = number of States
        self.M = b.shape[1] # M = number of symbols

    def fit(self, O: numpy.ndarray, max_iterations):
        for i in range(max_iterations):
            alpha = self.calc_alpha(O)
            beta = self.calc_beta(O)
            # xi[t, i, j] = Probability of being in state i in time t and state j in time t+1
            xi = self.calc_xi(O, alpha, beta)

            # Probability of being in state i in time t
            gamma = self.calc_gamma(alpha, beta)

            # tau[i, j] = expected number of transitions from i to j
            tau = numpy.sum(xi[:-1,:], axis=0)
            # taui = expected number of transitions from i
            taui = numpy.sum(gamma[:-1,:], axis=0)

            # nu[i] = expected number of times in state i
            nu = numpy.sum(gamma, axis=0, keepdims=True).T
            
            # Omega[i, k] = Expected number of times in state i and observing symbol k
            omega = numpy.zeros(shape=(self.N, self.M))

            # nu[i] = expected number of times in state j

            for i in range(self.N):
                for k in range(self.M):
                    for t in range(len(O)):
                        if O[t] == k:
                            omega[i,k] += gamma[t,i ]

            

            self.pi = gamma[0, :]
            self.a = tau / taui
            self.b = omega / nu 

            



    def calc_xi(self, O, alpha, beta):
        xi = numpy.zeros(shape=(len(O) - 1, len(self.a), len(self.a)))

        for t in range(len(O) - 1):
            for i in range(len(self.a)):
                xi[t, i, :] = alpha[t, i] * self.a[i, :] * self.b[:, O[t+1]] * beta[t+1, :]
            xi[t] = xi[t] / numpy.sum(xi[t])

        return xi

    # Listing 3.7
    # Rabiner (27)
    # gamma[i, t] = the probability of being in state S_i at time t
    # gamma.shape = (len(O), len(self.a))

    # def calc_gamma(self, xi):
    #     # Rabiner (38)
    #     gamma = numpy.sum(xi, axis=2)
    #     return gamma

    def calc_gamma(self, alpha: numpy.ndarray, beta: numpy.ndarray):
        # Rabiner (27)
        # gamma[i, t] = the probability of being in state i at time t
        product = alpha * beta
        norm = numpy.sum(product, axis=1, keepdims=True)
        gamma = product / norm 
        return gamma

    # Rabiner (29)
    # q[t] = The most likely State at time t
    def calc_q(self, gamma):
        q = numpy.argmax(gamma, axis=1)
        return q

    # Finds the best state sequence for a given observation sequence
    
    def viterbi2(self, O):
        pass
    
    # sigma[t, i] = probability of the most likely state sequence which ends in state i at time t
    def viterbi(self, O):
        # initialization:
        sigma = numpy.zeros(shape=(len(O), len(self.a)))
        sigma[0, :] = self.pi * self.b[:, O[0]]

        # normalize sigma?
        # sigma[0, :] = sigma[0, :] / sum(sigma[0,:])

        psi = numpy.zeros(shape=(len(O), len(self.a)))
        psi[0, :] = 0

        # Recursion:
        for t in range(1, len(O)):
            for i in range(len(self.a)):
                sigma[t, i] = numpy.max(sigma[t-1, :] * self.a[:, i]) * self.b[i, O[t]]
                psi[t, i] = numpy.argmax(sigma[t-1,:]*self.a[:,i])
        
        T = len(O) - 1

        q = numpy.zeros(len(O))
        q[T] = numpy.argmax(sigma[T, :])

        for t in range(T - 1, -1, -1):
            q[t] = psi[t+1, int(q[t+1])]

        return q

    def calc_beta(self, O: numpy.ndarray):
        T = len(O) - 1
        t = 0
        beta = numpy.zeros(shape=(T + 1, self.N))
        beta[T, :] = 1

        for t in range(T-1, -1, -1):
            for i in range(self.N):
                beta[t, i] = numpy.sum(beta[t+1,:] * self.a[i,:] * self.b[:, O[t+1]])
        
        return beta

    def calc_beta_scaled(self, O: numpy.ndarray):
        T = len(O) - 1
        beta = numpy.zeros(shape=(len(O), len(self.a)))
        beta[T ,:] = 1

        scalars = numpy.zeros(len(O))
        scalars[T] = 1 / numpy.sum(beta[T, :])

        for t in range(T - 1, -1, -1):
            for i in range(len(self.a)):
                beta[t, i] = numpy.sum(self.a[i, :] * self.b[:, O[t+1]] * beta[t+1,:])
            
            beta[t, :], scalars[t] = self.scale_probabilities(beta[t, :])

        return beta, scalars

    

    def scale_probabilities(self, vector):
        scalar = 1 / numpy.sum(vector)
        return vector*scalar, scalar

    def calc_alpha(self, O: numpy.ndarray):
        # Initialization:
        t = 0
        alpha = numpy.zeros(shape=(len(O), len(self.a)))
        alpha[t, :] = self.pi * self.b[:, O[t]]

        # Recursion:
        for t in range(1, len(O)):
            for i in range(self.N):
                S = numpy.sum(self.a[:, i] * alpha[t-1, :])
                alpha[t, i] = S * self.b[i, O[t]]

        # Unscaled Log Prob:
        # T = len(O) -1
        # P = numpy.sum(alpha[T,:])
        # log_P = numpy.log(P)

        return alpha

    def calc_expected_num_of_transitions_between(self, xi, O):
        # See Rabiner (39b)

        tau = numpy.sum(xi, axis=0)
        return tau

    def calc_expected_num_of_transitions_from(self, gamma):
       pass



    # See Listing 3.10
    def calc_alpha_scaled(self, O: numpy.ndarray):
        
        # Initialization:
        t = 0
        alpha = numpy.zeros(shape=(len(O), len(self.a)))
        alpha[t, :] = self.pi * self.b[:, O[t]]
        scalars = numpy.zeros(len(O))
        alpha[t, :], scalars[t] = self.scale_probabilities(alpha[t,:])

        # Recursion:
        for t in range(1, len(O)):
            for i in range(len(self.a)):
                s = numpy.sum(alpha[t-1, :] * self.a[:, i] )
                alpha[t, i] = self.b[i, O[t]] * s

            alpha[t, :], scalars[t] = self.scale_probabilities(alpha[t,:])
            
        return alpha, scalars

    def log_prob(self, O):
        _, scalars = self.calc_alpha_scaled(O)
        return - numpy.sum(numpy.log(scalars))




def main():
    n_states = 3
    n_symbols = 5

    pi = rand_stochastic_vector(n_states)
    b = rand_stochastic_matrix(n_states, n_symbols)
    a = rand_stochastic_matrix(n_states, n_states)

    T = 12
    O = npr.randint(low=0, high=n_symbols,  size=T)
    hmm = HMM(pi, b, a)
    prob = hmm.calc_alpha_scaled(O)
    print(prob)

if __name__ == '__main__':
    main()