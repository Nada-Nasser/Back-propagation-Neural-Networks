import numpy as np


def activation_function(v):
    return 1/(1+(np.exp(-v)))


def input_to_neuron(h , j , M , weights , x , a_prev):
    assert h > 0
    summation = 0
    print(M)
    for i in range(M):
        print("len = " , len(weights[h-1][j]))
        summation+= weights[h-1][j][i] * a_prev[0][i]


    print("SUMMM" , summation)
    return activation_function(summation)

def gradient_descent(x, y, M, H, Ls , N, K, weights, alpha, n_iterations):
    cost = np.zeros(n_iterations)  # list to store the cost in every iteration,

    for iteration in range(1):  # start the algorithm
        # 1- feedforword
        A = []
        A.append(x)
        n_As = 1
        # (a^h)j
        for h in range(1,H+1):
            print("HH" , h , " "  , H+1)
            for j in range(Ls[h]):
                print("JJ" , j , "  " , Ls[h])
                m = Ls[h-1]
                print( "a" , j , " = ", input_to_neuron(h, j, m, weights, x , A[n_As-1]))





