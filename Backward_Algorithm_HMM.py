import numpy as np

def backward(A, pi, B, observations):
    """
    A : Transient Matrix -> (N, N) Matrix 
    pi : Initial Probablities -> (N, ) Vector
    B : Emmision Matrix -> (N, T) Matrix
    observations : sequence of observations : a list with lenght, T
    """

    T = len(observations)
    N = pi.shape[0]
    
    """
    Creating beta matrix. 
    alpha is a (T, N) matrix.
    beta[t, :],the t'th row of the matrix, represents the P(ot+1, ot+2, ... ,oT , qt = Si)
    """
    beta = np.zeros((T, N))
    
    # Initialization
    '''
    Since There are no next states. the probablity of no next observation given any state is 1
    '''
    beta[T-1, :] = 1
    
    for t in range(0 , T-1)[::-1]: # iterate backward
        beta[t,:] = A @ (beta[t+1, :] * B[:,observations[t+1]])


    print(f"The beta matrix is : \n{beta}")
    return np.sum(beta[0,:] * pi * B[:,observations[0]])


def main():
    states = 0, 1, 2
    observations = [0, 0, 1]

    # Transition Probabilities
    A = np.array([[0.6, 0.2, 0.2], [0.5, 0.3, 0.2], [0.4, 0.1, 0.5]])

    # Initial Probabilities
    pi = np.array([0.5, 0.2, 0.3])

    # Emmision Probabilities
    B = np.array([[0.7, 0.1, 0.2], [0.1, 0.6, 0.3], [0.3, 0.3, 0.4]])

    prob = backward(A, pi, B, observations)
    print("Probability of the observed sequence is: ", np.round(prob, 4))

if __name__ == "__main__":
    main()