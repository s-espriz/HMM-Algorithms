import numpy as np
def Viterbi(A, pi, B, observations):
    """
    A : Transient Matrix -> (N, N) Matrix 
    pi : Initial Probablities -> (N, ) Vector
    B : Emmision Matrix -> (N, T) Matrix
    observations : sequence of observations : a list with lenght, T
    """

    T = len(observations)
    N = pi.shape[0]

    """
    initializing delta matrix
    delta_tj = max(P(q1, q2, ... , qt-1, qt = Si)) for (q1, q2, ... , qt-1)
    """
    delta = np.zeros((T, N))

    """
    let's construct the first row of the delta
    """
    delta[0, :] = pi * B[:, observations[0]]


    """
    initializing psi, which keeps the best path Matrix
    """
    psi = np.zeros((T, N))


    for t in range(1, T) : 
        delta[t, :] = np.max(delta[t-1 ,:] * A.T, axis = 1) * B[:, observations[t]]

        psi[t, :] = np.argmax(delta[t-1 ,:] * A.T, axis = 1)


    # finding best path based on calculated delta and psi
    best_path = np.zeros(T)
    best_path[T-1] = np.argmax(delta[T-1, :])
    for t in range(0 , T-1)[::-1]:
        best_path[t] = psi[t+1, int(best_path[t+1])]
    print(f"delta Matrix:\n{delta}\npsi = {psi}\nbest path = {best_path}")
    return best_path


def main():
    states = (0,1) 
    observations = [0,1,2]
    pi = np.array([0.6 , 0.4]) 

    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    B = np.array([
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]
    ])
    series_states = Viterbi(A, pi, B ,observations)
    print(f"the most likely series of Sates : {series_states}")
    
if __name__ == "__main__":
    main()