import numpy as np


def activation_function(v):
    return 1/(1+(np.exp(-v)))


def input_to_hidden_neuron(j, m, w1, x):
    summation = 0
    for i in range(m):
        summation += w1[j][i] * x[i]
    return activation_function(summation)


def compute_cost(N, ao, y):
    # 𝑬𝒌𝒐 = 𝟏/𝟐(𝒂𝒌𝒐 – 𝒚𝒌)𝟐
    error = 0
    for k in range(N):
        error += (1.0/2) * pow(ao[k] - y[k],2)
    return error


def input_to_output_neuron(k, L, w, ah):
    summation = 0
    for j in range(L):
        summation += w[k][j] * ah[j]
    return activation_function(summation)


def gradient_descent(X, Y, M, L, N, K, Wh, Wo, alpha, n_iterations):
    cost = []  # list to store the cost in every iteration,

    for iteration in range(n_iterations):  # start the algorithm
        for train in range(K):
            x = X[train]
            y = Y[train]

            # feedforward
            ah = []
            for j in range(L):  # For each hidden layer neuron j
                aj = input_to_hidden_neuron(j, M, Wh, x)
                ah.append(aj)

            ao = []
            for k in range(N):  # For each output layer neuron k
                ak = input_to_output_neuron(k, L, Wo, ah)
                ao.append(ak)

            # backpropagation
            delta_o = []
            for k in range(N):  # For each output neuron k:
                delta_k = (ao[k] - y[k]) * ao[k] *(1-ao[k])
                delta_o.append(delta_k)

            delta_h = []
            for j in range(L):  # For each hidden neuron j:
                # 𝛅𝒋𝒉 = (Σ 𝛅𝒌𝒐 ∗𝒘𝒌𝒋𝒐𝒏𝒌=𝟏)∗ 𝒂𝒋𝒉 ∗ (𝟏− 𝒂𝒋𝒉)
                term = 0
                for k in range(N):
                    term += delta_o[k] * Wo[k][j]
                delta_j = term * ah[j] * (1-ah[j])
                delta_h.append(delta_j)

            # update weights
            for k in range(N):
                for j in range(L):
                    # 𝒘𝒌𝒋𝒐 = 𝒘𝒌𝒋𝒐 – η * 𝛅𝒌𝒐 * 𝒂𝒋𝒉
                    Wo[k][j] = Wo[k][j] - alpha * delta_o[k] * ah[j]

            for j in range(L):
                for i in range(M):
                    # 𝒘𝒋𝒊𝒉 = 𝒘𝒋𝒊𝒉 – η * 𝛅𝒋𝒉 * 𝒙𝒊
                    Wh[j][i] = Wh[j][i] - alpha * delta_h[j] * x[i]

        cost.append(compute_cost(N,ao,y))

    return Wh, Wo, cost
