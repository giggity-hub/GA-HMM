import numpy
from typing import Tuple, NamedTuple, List
import numpy.random as npr
from numba import njit, jit
from hmm.types import MultipleObservationSequences


@njit
def calc_beta_scaled( O: numpy.ndarray, b:numpy.ndarray, a:numpy.ndarray, scalars: numpy.ndarray):
    N = b.shape[0]
    T = len(O)
    beta = numpy.zeros(shape=(T, N))
    beta[T-1, :] = scalars[T-1]


    for t in range(T - 2, -1, -1):
        for i in range(N):
            for j in range(N):
                beta[t, i] = beta[t,i] + a[i,j] * b[j, O[t+1]] * beta[t+1, j]
            
            beta[t,i] = scalars[t] * beta[t,i]

    return beta

@njit
def calc_alpha_scaled( O: numpy.ndarray, pi: numpy.ndarray, b:numpy.ndarray, a:numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    # Initialization:
    N = b.shape[0]
    t = 0
    alpha = numpy.zeros(shape=(len(O), N))
    scalars = numpy.zeros(len(O))

    alpha[0, :] = pi * b[:, O[0]]
    scalars[0] = 1 / numpy.sum(alpha[0,:])
    alpha[0, :] = alpha[0, :] * scalars[0]

    # Recursion:
    for t in range(1, len(O)):
        for i in range(N):
            for j in range(N):
                alpha[t,i] = alpha[t,i] + alpha[t-1, j] * a[j,i]

            alpha[t, i] = alpha[t,i] * b[i, O[t]]

        scalars[t] = 1 / numpy.sum(alpha[t,:])
        alpha[t, :] = alpha[t, :] * scalars[t]
    
    return alpha, scalars



@njit
def calc_gammas(b: numpy.ndarray, a: numpy.ndarray, alpha: numpy.ndarray, beta: numpy.ndarray, O: numpy.ndarray):
    T, N = alpha.shape

    di_gamma = numpy.zeros((T, N, N))
    gamma = numpy.zeros((T, N))

    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                di_gamma[t, i, j] = (alpha[t,i] * a[i,j] * b[j, O[t+1]] * beta[t+1, j])
                gamma[t, i] = gamma[t,i] + di_gamma[t, i, j]

    # Spezieller Käse
    for j in range(N):
        gamma[T-1, j] = alpha[T-1, j]

    return di_gamma, gamma

@njit
def calc_a_numer_and_denom(gamma: numpy.ndarray, di_gamma: numpy.ndarray):
    T, N = gamma.shape

    numer = numpy.zeros((N, N))
    denom = numpy.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # numer = 0
            # denom = 0

            for t in range(T-1):
                numer[i,j] += di_gamma[t, i, j]
                denom[i,j] += gamma[t, i]

            # a[i, j] = numer / denom
    
    return numer, denom

@njit
def calc_b_numer_and_denom(gamma: numpy.ndarray, O: numpy.ndarray, M: int):
    T, N = gamma.shape 

    numer = numpy.zeros((N, M))
    denom = numpy.zeros((N, M))

    for i in range(N):
        for j in range(M):
            for t in range(T):
                if O[t] == j:
                    numer[i, j] += gamma[t, i]
                
                denom[i, j] += gamma[t, i]

    return numer, denom

@njit
def reestimate_single_observation(pi, b, a, O):

    
    alpha, scalars = calc_alpha_scaled(O, pi, b, a)
    beta = calc_beta_scaled(O, b, a, scalars)

    di_gamma, gamma = calc_gammas(b, a, alpha, beta, O)

    pi = gamma[0, :]

    numer_a, denom_a = calc_a_numer_and_denom(gamma, di_gamma)
    numer_b, denom_b = calc_b_numer_and_denom(gamma, O, M=b.shape[1])
    a = numer_a / denom_a
    b = numer_b / denom_b

    log_prob = - numpy.sum(numpy.log(scalars))

    return pi, b, a, log_prob

@njit
def reestimate_multiple_observations(
    pi: numpy.ndarray, 
    b: numpy.ndarray, 
    a:numpy.ndarray, 
    all_observations: MultipleObservationSequences):

    slices, arrays, length = all_observations
        
    numer_a_total = numpy.zeros(a.shape)
    denom_a_total = numpy.zeros(a.shape)
    numer_b_total = numpy.zeros(b.shape)
    denom_b_total = numpy.zeros(b.shape)
    pi_total = numpy.zeros(pi.shape)

    log_prob_total = 0

    for i in range(length):
        start = slices[i]
        stop = slices[i+1]
        O = arrays[start: stop]

        alpha, scalars = calc_alpha_scaled(O, pi, b, a)
        beta = calc_beta_scaled(O, b, a, scalars)
        di_gamma, gamma = calc_gammas(b, a, alpha, beta, O)

        pi = gamma[0, :]
        pi_total += pi

        numer_a, denom_a = calc_a_numer_and_denom(gamma, di_gamma)
        numer_a_total = numer_a_total + numer_a
        denom_a_total = denom_a_total + denom_a

        numer_b, denom_b = calc_b_numer_and_denom(gamma, O, M=b.shape[1])
        numer_b_total = numer_b_total + numer_b
        denom_b_total = denom_b_total + denom_b
        
        log_prob = - numpy.sum(numpy.log(scalars))
        log_prob_total += log_prob


    pi = pi_total / length
    a = numer_a_total / denom_a_total
    b = numer_b_total / denom_b_total

    return pi, b, a, log_prob_total



@njit
def calc_log_prob(pi, b, a, O:numpy.ndarray):
    alpha, scalars = calc_alpha_scaled(O, pi, b, a)
    log_prob = -numpy.sum(numpy.log(scalars))
    return log_prob










    
    



