import numpy
from typing import Tuple, NamedTuple, List
from lib.utils import rand_stochastic_matrix, rand_stochastic_vector
import numpy.random as npr
from numba import njit, jit
from hmm.types import MultipleObservationSequences, HmmParams



def multiple_observation_sequences_from_ndarray_list(ndarray_list: List[numpy.ndarray]) -> MultipleObservationSequences:
    """Concatenates a list of 1-D arrays and also remembers their slices

    Args:
        ndarray_list (List[numpy.ndarray]): _description_

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The first return Value is the concatenated array. The second value
        contains the slices so that ndarray_list[i] == unified_array[indices[i]:indices[i+1]]
    """
    indices = numpy.zeros((len(ndarray_list) + 1), dtype=int)

    for i in range(len(ndarray_list)):
        indices[i + 1] = indices[i] + len(ndarray_list[i])
    
    unified_array = numpy.concatenate(ndarray_list)
    
    return MultipleObservationSequences(slices=indices, arrays=unified_array, length=len(ndarray_list))



@njit(inline='always')
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
                beta[t, i] = beta[t,i] + a[i,j] * b[j, O[t+1]] * beta[t+1, j]
            
            # scalars[t] = 1 / numpy.sum(beta[t, :])
            beta[t,i] = scalars[t] * beta[t,i]

    return beta

@njit(inline='always')
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
                alpha[t,i] = alpha[t,i] + alpha[t-1, j] * a[j,i]

            alpha[t, i] = alpha[t,i] * b[i, O[t]]

        scalars[t] = 1 / numpy.sum(alpha[t,:])
        for i in range(N):
            alpha[t, i] = scalars[t] * alpha[t, i]
    
    return alpha, scalars

@njit(inline='always')
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

@njit(inline='always')
def calc_gamma(xi, alpha):
    T = xi.shape[0]
    gamma = numpy.sum(xi, axis=2)
    gamma[T-1, :] = alpha[T-1,:]
    return gamma




@njit(inline='always')
def reestimate(pi, b, a, O):
    alpha, scalars = calc_alpha_scaled(O, pi, b, a)
    beta = calc_beta_scaled(O, b, a, scalars)

    xi = calc_xi(a, b, alpha, beta, O)
    gamma = calc_gamma(xi, alpha)
    # gammai = calc_gammai(gamma)


    n_states, n_symbols = b.shape

    for i in range(n_states):
        pi[i] = gamma[0, i]

    denom_a = numpy.sum(gamma[:(len(O)-1), :], axis=0).reshape((n_states, 1))
    numer_a = numpy.sum(xi, axis=0)
    a = numer_a / denom_a

    denom_b = numpy.sum(gamma, axis=0).reshape((n_states, 1))
    numer_b = calc_omega(gamma, O, n_symbols)

    b = numer_b / denom_b
    # reestimate B
    
    
    log_prob = - numpy.sum(numpy.log(scalars))

    return pi, b, a, log_prob

@njit(inline='always')
def reestimate_multiple_observations(pi: numpy.ndarray, b: numpy.ndarray, a:numpy.ndarray, all_observations: MultipleObservationSequences):
    slices, arrays, length = all_observations
        
    numer_a_total = numpy.zeros(a.shape)
    denom_a_total = numpy.zeros(a.shape[0])
    numer_b_total = numpy.zeros(b.shape)
    denom_b_total = numpy.zeros(b.shape[0])
    pi_total = numpy.zeros(pi.shape)

    n_states, n_symbols = b.shape

    log_prob_total = 0

    for i in range(length):
        start = slices[i]
        stop = slices[i+1]

        O = arrays[start: stop]
        # O = Obs[i]
        # import ipdb; ipdb.set_trace()
        alpha, scalars = calc_alpha_scaled(O, pi, b, a)
        beta = calc_beta_scaled(O, b, a, scalars)
        xi = calc_xi(a, b, alpha, beta, O)
        gamma = calc_gamma(xi, alpha)

        
        denom_a_total = denom_a_total + numpy.sum(gamma[:(len(O)-1), :], axis=0)#.reshape((n_states, 1))
        numer_a_total = numer_a_total +  numpy.sum(xi, axis=0)
        

        denom_b_total = denom_b_total + numpy.sum(gamma, axis=0)#.reshape((n_states, 1))
        numer_b_total = numer_b_total +  calc_omega(gamma, O, n_symbols)

        pi_total = pi_total + gamma[0,:]
        
        
        log_prob_total -= numpy.sum(numpy.log(scalars))


    pi = pi_total / length
    a = numer_a_total / numpy.atleast_2d(denom_a_total).T
    b = numer_b_total / numpy.atleast_2d(denom_b_total).T

    return pi, b, a, log_prob_total



@njit(inline='always')
def calc_omega(gamma: numpy.ndarray, O: numpy.ndarray, n_symbols: int):
    n_states = gamma.shape[1]
    omega = numpy.zeros((n_states, n_symbols))
    for i in range(n_states):
        for k in range(n_symbols):
            for t in range(len(O)):
                if O[t] == k:
                    omega[i, k] = omega[i,k] + gamma[t, i]
            
            # b[i, j] = numer / denom
    return omega


@njit(inline='always')
def train(pi: numpy.ndarray, b: numpy.ndarray, a:numpy.ndarray, all_observations: MultipleObservationSequences, n_iterations: int=1):
    
    for i in range(n_iterations):
        pi, b, a, log_prob_total = reestimate_multiple_observations(pi, b, a, all_observations )
    return pi, b, a, log_prob_total

@njit(inline='always')
def calc_log_prob(pi, b, a, O:numpy.ndarray):
    alpha, scalars = calc_alpha_scaled(O, pi, b, a)
    log_prob = -numpy.sum(numpy.log(scalars))
    return log_prob

@njit(inline='always')
def calc_total_log_prob(hmm_params: HmmParams, all_observation_seqs: MultipleObservationSequences):
    pi, b, a = hmm_params
    total_log_prob = 0

    slices, array, length = all_observation_seqs

    for i in range(length):
        start = slices[i]
        stop = slices[i+1]

        single_observation_seq = array[start:stop]
        log_prob = calc_log_prob(pi, b, a, single_observation_seq)
        total_log_prob += log_prob
    
    return total_log_prob

def calc_mean_log_prob(hmm_params: HmmParams, all_observation_seqs: MultipleObservationSequences):
    total_log_prob = calc_total_log_prob(hmm_params, all_observation_seqs)
    mean_log_prob = total_log_prob / all_observation_seqs.length
    return mean_log_prob







    
    



