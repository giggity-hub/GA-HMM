import numpy
from typing import Tuple, NamedTuple, List
from lib.utils import rand_stochastic_matrix, rand_stochastic_vector
import numpy.random as npr
from numba import njit, jit

class HmmParams(NamedTuple):
    start_vector: numpy.ndarray
    emission_matrix: numpy.ndarray
    transition_matrix: numpy.ndarray

@njit
def calc_beta_scaled( O: numpy.ndarray, b:numpy.ndarray, a:numpy.ndarray, scalars: numpy.ndarray):
    N = b.shape[0]
    T = len(O) - 1
    beta = numpy.zeros(shape=(len(O), N))
    for i in range(N):
        beta[T, i] = scalars[T]


    for t in range(T - 1, -1, -1):
        for i in range(N):
            beta[t, i] = 0
            for j in range(N):
                beta[t, i] += a[i,j] * b[j, O[t+1]] * beta[t+1, j]
            
            # scalars[t] = 1 / numpy.sum(beta[t, :])
            beta[t,i] = scalars[t] * beta[t,i]

    return beta

@njit
def calc_alpha_scaled( O: numpy.ndarray, pi: numpy.ndarray, b:numpy.ndarray, a:numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Initialization:
    N = b.shape[0]
    t = 0
    alpha = numpy.zeros(shape=(len(O), N))
    scalars = numpy.zeros(len(O))

    for i in range(N):
        alpha[0, i] = pi[i] * b[i, O[0]]
    

    scalars[t] = 1 / numpy.sum(alpha[0,:])
    for i in range(N):
        alpha[0, i] = scalars[0] * alpha[0, i]

    # Recursion:
    for t in range(1, len(O)):
        for i in range(N):
            alpha[t, i] = 0
            for j in range(N):
                alpha[t,i] += alpha[t-1, j] * a[j,i]

            alpha[t, i] = alpha[t,i] * b[i, O[t]]

        scalars[t] = 1 / numpy.sum(alpha[t,:])
        for i in range(N):
            alpha[t, i] = scalars[t] * alpha[t, i]
    
    return alpha, scalars

@njit
def calc_xi(a, b, alpha: numpy.ndarray, beta: numpy.ndarray, O):
    # Rabiner (27)
    # gamma[i, t] = the probability of being in state i at time t
    # product = alpha * beta
    # norm = numpy.sum(product, axis=1, keepdims=True)
    # gamma = product / norm 
    T = len(O)
    N = alpha.shape[1]
    xi = numpy.zeros((T, N, N))
    # gamma = numpy.zeros((T, N))
    for t in range(T -1):
        for i in range(N):
            # gamma[t, i] = 0
            for j in range(N):
                xi[t, i, j] = alpha[t, i] * a[i,j] * b[j, O[t+1]] * beta[t+1, j]
                # gamma[t, i] += xi[t, i, j]

    # for i in range(N):
    #     gamma[T-1, i] = alpha[T-1, i]
    return xi

@njit
def calc_gamma(xi, alpha):
    T = xi.shape[0]
    gamma = numpy.sum(xi, axis=2)
    gamma[T-1, :] = alpha[T-1,:]
    return gamma

def calc_mean_log_prob(hmm_params: HmmParams, all_observation_seqs: List[numpy.ndarray]):
    pi, b, a = hmm_params
    total_log_prob = 0
    for observation_seq in all_observation_seqs:
        _, log_prob = calc_alpha_scaled(observation_seq, pi, b, a)
        total_log_prob += log_prob
    
    n_seqs = len(all_observation_seqs)
    mean_log_prob = total_log_prob / n_seqs
    return mean_log_prob


@njit
def reestimate(pi, b, a, O):
    alpha, scalars = calc_alpha_scaled(O, pi, b, a)
    beta = calc_beta_scaled(O, b, a, scalars)

    xi = calc_xi(a, b, alpha, beta, O)
    gamma = calc_gamma(xi, alpha)
    # gammai = calc_gammai(gamma)


    n_states, n_symbols = b.shape

    for i in range(n_states):
        pi[i] = gamma[0, i]

    # reestimate A
    for i in range(n_states):
        denom = 0
        for t in range(len(O) -1):
            denom += gamma[t, i]
        
        for j in range(n_states):
            numer = 0
            for t in range(len(O) -1):
                numer += xi[t, i, j]

            a[i,j] = numer / denom

    # reestimate B
    for i in range(n_states):
        denom = 0
        for t in range(len(O)):
            denom += gamma[t,i]
        
        for j in range(n_symbols):
            numer = 0 
            for t in range(len(O)):
                if O[t] == j:
                    numer += gamma[t, i]
            
            b[i, j] = numer / denom
    
    log_prob = - numpy.sum(numpy.log(scalars))

    return pi, b, a, log_prob


def baum_welch(hmm_params: HmmParams, observation_sequences, n_iterations=1) -> HmmParams:
    pi, b, a = hmm_params
    n_states, n_symbols = b.shape

    log_prob_sum = 0
    gamma_sum = 0
    tau_sum = 0
    taui_sum = 0
    nu_sum = 0
    omega_sum = 0

    for O in observation_sequences:
        alpha, scalars = calc_alpha_scaled(O, pi, b, a)
        beta = calc_beta_scaled(O, b, a, scalars)

        # Probability of being in state i in time t
        gamma = calc_gamma(alpha, beta)

        # xi[t, i, j] = Probability of being in state i in time t and state j in time t+1
        xi = calc_xi_from_gamma(gamma, O)

        # tau[i, j] = expected number of transitions from i to j
        tau = calc_tau(xi)

        # taui[i] = expected number of transitions from i
        taui = calc_taui(tau)

        # nu[i] = expected number of times in state i
        nu = calc_nu(gamma)
        
        # Omega[i, k] = Expected number of times in state i and observing symbol k
        omega = calc_omega(gamma, O, n_states, n_symbols)

        # Increment all those motherfuckers

        gamma_sum += gamma
        tau_sum += tau
        taui_sum += taui
        nu_sum += nu
        omega_sum += omega

    n_observation_sequences = len(observation_sequences)

    pi = gamma_sum[0,:] / n_observation_sequences
    b = omega_sum / nu_sum
    a = tau_sum / taui_sum

    return HmmParams(pi, b, a)





    
    



