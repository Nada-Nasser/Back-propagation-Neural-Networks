import numpy as np


def activation_function(v):  # use the sigmoid function as an activation function
    return 1/(1+(np.exp(-v)))


def feed_forward(j, m, w, x):  # feed forward
    summation = 0
    for i in range(m):  # for each neuron i in the layer (hidden/ output)
        summation += w[j][i] * x[i]
    return activation_function(summation)  # apply the activation function on the input to the layer (hidden/ output)


def compute_cost(N, ao, y):
    # 𝑬𝒌𝒐 = 𝟏/𝟐(𝒂𝒌𝒐 – 𝒚𝒌)𝟐
    error = 0
    for k in range(N):  # for each neuron K in output layer.
        error += (1.0/2) * pow(ao[k] - y[k], 2)  # Compute the error for each neuron in the output layer.
    return error


def gradient_descent(X, Y, M, L, N, K, Wh, Wo, alpha, n_iterations):
    cost = []  # list to store the cost in every iteration,

    for iteration in range(n_iterations):  # start the algorithm
        c = 0
        for train in range(K):
            x = X[train]
            y = Y[train]

            # feedforward
            ah = []  # output of each neuron in the hidden layer
            for j in range(L):  # For each hidden layer neuron j
                # use feed_forward to get the output of the hidden layer using x(training example)
                # as input to this layer
                aj = feed_forward(j, M, Wh, x)
                ah.append(aj)

            ao = []  # output of each neuron in the out layer
            for k in range(N):  # For each output layer neuron k
                # use feed_forward to get the output of the out layer using [ah] as input to this layer
                ak = feed_forward(k, L, Wo, ah)
                ao.append(ak)

            # backpropagation
            delta_o = [] # 𝛅 (the product of the error) for each neuron in out layer
            for k in range(N):  # For each output neuron k:
                #  𝛅𝒐[𝒌] = (𝒂𝒐[𝒌] – 𝒚𝒌) ∗ 𝒂𝒐[𝒌] ∗ (𝟏 − 𝒂𝒐[𝒌])
                delta_k = (ao[k] - y[k]) * ao[k] *(1-ao[k])
                delta_o.append(delta_k)

            delta_h = []  # 𝛅 (the product of the error) for each neuron in hidden layer
            for j in range(L):  # For each hidden neuron j:
                # 𝛅𝒋𝒉 = (Σ 𝛅𝒌𝒐 ∗ 𝒘𝒌𝒋𝒐𝒏𝒌=𝟏)∗ 𝒂𝒋𝒉 ∗ (𝟏− 𝒂𝒋𝒉)
                term = 0
                for k in range(N):
                    term += delta_o[k] * Wo[k][j]
                delta_j = term * ah[j] * (1-ah[j])
                delta_h.append(delta_j)

            # update weights
            for k in range(N):  # For each output neuron k:
                for j in range(L):  # For each hidden neuron j:
                    # 𝒘𝒌𝒋𝒐 = 𝒘𝒌𝒋𝒐 – η * 𝛅𝒌𝒐 * 𝒂𝒋𝒉
                    Wo[k][j] = Wo[k][j] - (alpha * delta_o[k] * ah[j])  # update the weight[k][j] in the output layer

            for j in range(L):  # For each hidden neuron j:
                for i in range(M):  # For each input neuron i:
                    # 𝒘𝒋𝒊𝒉 = 𝒘𝒋𝒊𝒉 – η * 𝛅𝒋𝒉 * 𝒙𝒊
                    Wh[j][i] = Wh[j][i] - (alpha * delta_h[j] * x[i])  # update the weight[j][i] in the hidden layer

            c += compute_cost(N, ao, y)  # compute cost for current training example x

        cost.append(c/K)

    return Wh, Wo, cost
