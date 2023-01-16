import numpy
from typing import Tuple
from lib.utils import rand_stochastic_matrix, rand_stochastic_vector
import numpy.random as npr




class HMM:
    def __init__(self, pi: numpy.ndarray, b: numpy.ndarray, a: numpy.ndarray) -> None:
        self.pi = pi 
        self.a = a 
        self.b = b

    def fit(self, O: numpy.ndarray, max_iterations):
        for i in range(max_iterations):
            alpha, c = self.calc_alpha(O)
            beta, v = self.calc_beta(O)

            gamma = self.calc_gamma(alpha, beta)
            xi = self.calc_xi(O, alpha, beta)
            
            self.a = gamma[1, :]
        
            self.a = numpy.sum(xi, axis=0) / numpy.sum(gamma, axis=0)

            for i in range(len(self.a)):
                
                for k in range(self.b.shape[1]):
                    shish = 0
                    saas = 0
                    for t in range(len(O)):
                        shish += gamma[t, i] * (1 if O[t] == k else 0)
                        saas += gamma[t, i]


                    self.b[i, k] = shish / saas

        # print(self.a)
        # print(self.b)
        # print(self.pi)

    
    def calc_xi(self, O, alpha, beta):
        xi = numpy.zeros(shape=(len(O) - 1, len(self.a), len(self.a)))

        for t in range(len(O) - 1):
            for i in range(len(self.a)):
                # for j in range(len(self.a)):
                xi[t, i, :] = alpha[t, i] * self.a[i, :] * self.b[:, O[t+1]] * beta[t+1, :]
            xi[t] = xi[t] / numpy.sum(xi[t])

        return xi

    # Listing 3.7
    # Rabiner (27)
    # gamma[i, t] = the probability of being in state S_i at time t
    # gamma.shape = (len(O), len(self.a))
    def calc_gamma(self, alpha: numpy.ndarray, beta: numpy.ndarray):
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
        beta = numpy.zeros(shape=(len(O), len(self.a)))
        beta[T ,:] = 1

        v = numpy.zeros(len(O))
        v[T] = 1 / numpy.sum(beta[T, :])

        for t in range(T - 1, -1, -1):
            for i in range(len(self.a)):
                beta[t, i] = numpy.sum(self.a[i, :] * self.b[:, O[t+1]] * beta[t+1,:])
            
            v[t] = 1 / numpy.sum(beta[t, :])
            beta[t, :] = v[t] * beta[t, :]

        return beta, v


    # See Listing 3.10
    def calc_alpha(self, O: numpy.ndarray):

        # # fixed_t
        # max_t = 10
        
        # # initialization
        # alpha = numpy.zeros(shape=(len(O), len(self.a)))
        # alpha[0, :] = self.pi * self.b[:, O[0]]

        # c1 = numpy.zeros(len(O))
        # c1[0] = 1 / numpy.sum(alpha[0, :])
        # # res = numpy.log(numpy.sum(alpha[0,:]))

        # for t in range(1, len(O)):
            
        #     for j in range(len(self.a)):
        #         # Induction Step
        #         alpha[t, j] = numpy.sum(alpha[t-1, :] * self.a[:, j]) * self.b[j, O[t]]

        #     c1[t] = (c1[t-1]) * (1 / numpy.sum(alpha[t,:]))

        # print(c1)
        # print(numpy.log(c1))

        alpha_scaled = numpy.zeros(shape=(len(O), len(self.a)))
        c2 = numpy.zeros(len(O))

        alpha_scaled[0, :] = self.pi * self.b[:, O[0]]

        c2[0] = 1 / numpy.sum(alpha_scaled[0, :])
        alpha_scaled[0, :] = alpha_scaled[0, :] * c2[0]

        for t in range(1, len(O)):
            for i in range(len(self.a)):
                s = numpy.sum(alpha_scaled[t-1, :] * self.a[:, i] )
                alpha_scaled[t, i] = self.b[i, O[t]] * s

            c2[t] = 1 / numpy.sum(alpha_scaled[t, :])
            alpha_scaled[t, :] = alpha_scaled[t, :] * c2[t]
            
        return alpha_scaled, c2


    def alpha_scaled(self, t, i, O, alpha_scaled):
        pass

    def log_prob(self, O):
        _, c2 = self.calc_alpha(O)
        return - numpy.sum(numpy.log(c2))

def main():
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
    main()