import numpy as np

def forward_algorithm(A:np.ndarray,B:np.ndarray,O:np.ndarray,I:np.ndarray):
    """_summary_

    Args:
        A (np.ndarray): State Transition Matrix
        B (_type_): Confusion Matrix
        O (_type_): Observation Vector
        I (_type_): Initial Probabilities Vector
    """

    m,n = B.shape
    N=len(O)

    # Initialization
    Alfa=np.zeros((N,m))
    Alfa[0,:] = I * B[:, O[0]]

    # Recursion
    for l in range(1,N):
        for k in range(m):
            S = 0
            for i in range(m):
                S=S+A[i,k]*Alfa[l-1,i]
            
            Alfa[l,k]=B[k, O[l]]*S

    # Probability of observing O
    P = np.sum(Alfa[N-1,:])

    return Alfa, np.log(P)

def backward_algorithm(A:np.ndarray,B:np.ndarray,O:np.ndarray):
    m, n = B.shape
    N = len(O)

    # Initialization
    Beta=np.zeros((N,m))
    Beta[N-1,:] = 1

    # Recursion
    for t in range(N-2, -1, -1):
        for i in range(m):
            Beta[t,i] = 0
            for j in range(m):
                Beta[t,i] += A[i,j] * B[j, O[t+1]] * Beta[t+1,j]
    
    return Beta


def viterbi_algorithm(A:np.ndarray,B:np.ndarray,O:np.ndarray,c:np.ndarray):
    """_summary_

    Args:
        A (np.ndarray): State Transition Matrix
        B (_type_): Confusion Matrix
        O (_type_): Observation Vector
        c (_type_): 1xm initial probabilities vector
    """
    
    m,n = B.shape
    N=len(O)

    # Initialization
    delta= np.zeros((N,m))
    phi = np.zeros((N,m))
    t=0
    delta[t,:] = c * B[O[t], :]
    phi[t, :] = 0

    # Recursion
    for t in range(1, N):
        for k in range(m):
            tmp = np.zeros(m)
            for l in range(m):
                tmp[l]=delta[t-1, l] * A[l, k] * B[k, O[t]]
            
            delta[t,k] = np.max(tmp)
            phi[t,k] = np.argmax(tmp)

    # Path finding
    q=np.zeros(N)
    Inx = np.argmax(delta[N-1,:])
    q[N-1] = Inx
    for k in range(N-2, -1, -1):
        q[k] = phi[k+1, q[k+1]]
    
    return q

# IST NICHT GETREU IMPLEMENTIERT
def compute_nu(Gama:np.ndarray,B:np.ndarray):
    """_summary_
        Return the number of visits to state i
        m hidden states, n output states and N observations
    Args:
        Gama (np.ndarray): _description_
        B (np.ndarray): Confusion Matrix
    """

    _, n = B.shape
    nu = np.sum(Gama, axis=0)

    return nu




