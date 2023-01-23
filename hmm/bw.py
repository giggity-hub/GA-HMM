import numpy
from typing import Tuple
from lib.utils import rand_stochastic_matrix, rand_stochastic_vector
import numpy.random as npr

def calc_beta(self, O: numpy.ndarray):
    T = len(O) - 1
    t = 0
    beta = numpy.zeros(shape=(T + 1, self.N))
    beta[T, :] = 1

    for t in range(T-1, -1, -1):
        for i in range(self.N):
            beta[t, i] = numpy.sum(beta[t+1,:] * self.a[i,:] * self.b[:, O[t+1]])
    
    return beta

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

class BaumWelch:
    def __init__(self, pi: numpy.ndarray, b: numpy.ndarray, a: numpy.ndarray) -> None:
        self.pi = pi 
        self.a = a 
        self.b = b
        self.N = b.shape[0] # N = number of States
        self.M = b.shape[1] # M = number of symbols

    
    

    def calc_beta_scaled(self, O: numpy.ndarray):

        T = len(O) - 1
        beta = numpy.zeros(shape=(len(O), self.N))
        beta[T ,:] = 1

        scalars = numpy.zeros(len(O))
        scalars[T] = 1 / numpy.sum(beta[T, :])

        for t in range(T - 1, -1, -1):
            for i in range(self.N):
                beta[t, i] = numpy.sum(self.a[i, :] * self.b[:, O[t+1]] * beta[t+1,:])
            
            scalars[t] = 1 / numpy.sum(beta[t,:])
            beta[t,:] = scalars[t] * beta[t,:]

        return beta

    # See Listing 3.10
    def calc_alpha_scaled(self, O: numpy.ndarray) -> Tuple[numpy.ndarray, float]:
        
        # Initialization:
        t = 0
        alpha = numpy.zeros(shape=(len(O), self.N))
        
        alpha[t, :] = self.pi * self.b[:, O[t]]
        scalars = numpy.zeros(len(O))
        scalars[t] = 1 / numpy.sum(alpha[t,:])
        alpha[t,:] = scalars[t] * alpha[t,:]

        # Recursion:
        for t in range(1, len(O)):
            for i in range(self.N):
                s = numpy.sum(alpha[t-1, :] * self.a[:, i] )
                alpha[t, i] = self.b[i, O[t]] * s

            scalars[t] = 1 / numpy.sum(alpha[t,:])
            alpha[t,:] = scalars[t] * alpha[t,:]

        log_probability = - numpy.sum(numpy.log(scalars))    
        return alpha, log_probability

    def calc_xi_from_gamma(self, gamma, O):
        xi = numpy.zeros((len(O), self.N, self.N))
        for t in range(len(O)-1):
            for i in range(self.N):
                for j in range(self.N):
                    xi[t, i, j] = gamma[t, i] * gamma[t+1, j]
        
        return xi


    def train(self, O, n_iterations=1):
        for i in range(n_iterations):
            self.reestimate(O)
        

    def reestimate(self, Observations: numpy.ndarray, max_iterations=1):
        log_prob_sum = 0
        gamma_sum = 0
        tau_sum = 0
        taui_sum = 0
        nu_sum = 0
        omega_sum = 0

        for O in Observations:
            alpha, log_prob = self.calc_alpha_scaled(O)
            beta = self.calc_beta_scaled(O)

            # Probability of being in state i in time t
            gamma = self.calc_gamma(alpha, beta)

            # xi[t, i, j] = Probability of being in state i in time t and state j in time t+1
            xi = self.calc_xi_from_gamma(gamma, O)

            # tau[i, j] = expected number of transitions from i to j
            tau = numpy.sum(xi, axis=0)

            # taui[i] = expected number of transitions from i
            taui = numpy.sum(tau, axis=1)

            # nu[i] = expected number of times in state i
            nu = numpy.sum(gamma, axis=0, keepdims=True).T
            
            # Omega[i, k] = Expected number of times in state i and observing symbol k
            omega = self.calc_omega(gamma, O)

            # Increment all those motherfuckers
            log_prob_sum += log_prob
            gamma_sum += gamma
            tau_sum += tau
            taui_sum += taui
            nu_sum += nu
            omega_sum += omega
            
        self.alpha = alpha 
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.taui = taui
        self.nu = nu
        self.omega = omega

        n_observation_sequences = len(Observations)

        self.pi = gamma_sum[0,:] / n_observation_sequences
        self.a = tau_sum / taui_sum
        self.b = omega_sum / nu_sum

        return log_prob_sum

    def calc_omega(self, gamma, O):
        omega = numpy.zeros(shape=(self.N, self.M))
        for i in range(self.N):
                for k in range(self.M):
                    for t in range(len(O)):
                        if O[t] == k:
                            omega[i,k] += gamma[t,i ]
        return omega


    def calc_tau_from_gamma(self, gamma, O):
        tau = numpy.zeros(shape=(self.N, self.N))

        for t in range(len(O)-1):
            for i in range(self.N):
                for j in range(self.N):
                    tau[i,j] += gamma[t, i] * gamma[t+1,j]
        return tau


            
    def calc_tau(self, alpha, beta, O):
        
        tau= numpy.zeros(shape=(self.N, self.N))
        for t in range(len(O)-1):
            num = (self.a * (alpha[t,:].reshape((self.N,1)) *beta[t+1,:].reshape((1,self.N)))) * ((self.b[:, O[t+1]].reshape((self.M, 1))*numpy.ones((1,self.N))).T)
            den= numpy.sum(num)
            tau=tau+num/den
            # for i in range(self.N):
            #     for j in range(self.N):
            #         tau[i,j] += alpha[t, i]*self.a[i,j] * beta[t+1, j] *self.b[j,O[t+1]]


            # for i in range(self.N):
            #     tau[i,:] = alpha[t, i] * self.a[i, :] * self.b[:, O[t+1]] * beta[t+1, :]
            #     tau[i,:] = tau[i,:] / sum(tau[i,:])
            
            # num=self.a*(alpha[t,:].reshape(-1,1) * beta[t+1,:])*(self.b[:,O[t+1]] *numpy.ones((1,self.N)))
            # den = numpy.ones((1,self.N)) *num * numpy.ones((self.N,1))
            # tau = tau+num/den
            # num = self.a * (alpha[t,:].T * beta[t+1,:]) * ((self.b[:,O[t +1]]*numpy.ones((1,self.N))).T)
            # den = numpy.ones((1,self.N))*num*numpy.ones((self.N,1))
            # den = sum(num)
            # tau = tau+num/den
        return tau

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

    def calc_log_prob(self, observation_seq):
        _, log_probability = self.calc_alpha_scaled(observation_seq)
        return log_probability

    def calc_mean_log_prob(self, all_observation_seqs):
        total_log_prob = 0
        for observation_seq in all_observation_seqs:
            _, log_prob = self.calc_alpha_scaled(observation_seq)
            total_log_prob += log_prob
        
        n_seqs = len(all_observation_seqs)
        mean_log_prob = total_log_prob / n_seqs
        return mean_log_prob
    




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