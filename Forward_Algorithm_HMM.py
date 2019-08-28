import numpy as np

def forward(A, pi, B, observations):
    M = len(observations)
    N = pi.shape[0]
    
    alpha = np.zeros((M, N))
    
    # Initialization
    '''
    Since their are no previous states. the probability of being in state 1 at time 1 is given as product of:
    1. Initial Probability of being in state 1
    2. Emmision Probability of symbol O(1) being in state 1
    '''
    alpha[0, :] = pi * B[:,observations[0]]
    
    # Induction
    '''
    if we know the previous state i,then the probability of being in state j at time t+1 is given as product of:
    1. Probability of being in state i at time t
    2. Transition probability of going from state i to state j
    3. Emmision Probability of symbol O(t+1) being in state j
    '''
    for t in range(1, M):
        for j in range(N):
            for i in range(N):
                alpha[t, j] += alpha[t-1, i] * A[i, j] * B[j, observations[t]]
    
    return np.sum(alpha[M-1,:])


def main():
    states = 0, 1, 2
    observations = [0, 0, 1]

    # Transition Probabilities
    A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])

    # Initial Probabilities
    pi = np.array([0.5, 0.2, 0.3])

    # Emmision Probabilities
    B = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])

    prob = forward(A, pi, B, observations)
    print("Probability of the observed sequence is: ", prob)

if __name__ == "__main__":
    main()