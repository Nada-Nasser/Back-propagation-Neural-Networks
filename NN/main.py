import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import *

line = np.array(pd.read_csv("train.txt", header=None, delim_whitespace=True, nrows=2))
train_data = pd.read_csv("train.txt", skiprows=[0, 1], header=None, delim_whitespace=True)
M, L, N, K = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[1][0])

print("M = ", M)  # N is number of Input Nodes
print("L = ", L)  # L is number of Hidden Nodes
print("N = ", N)  # N is number of Output Nodes
print("K = ", K)  # K, the number of training examples
# each line has length M+N values,
# first M values are X vector
# last N values are output values.

# separate X (training data) from y (target variable)
cols = train_data.shape[1]
X = train_data.iloc[:, 0: int(M)]
Y = train_data.iloc[:, int(M):]

# rescaling data
X = (X - X.mean()) / X.std()

# add ones column
X.insert(0, 'Ones', 1)
X = np.array(X)
print(X)

M = M + 1
# weights = np.array([0] * 2)
#
# weights[0] = np.array([[0]*M for i in range(L)])
# weights[1] = np.array([[0]*L for i in range(N)])

w1 = np.array([[0]*M for i in range(L)])
w2 = np.array([[0]*L for i in range(N)])
weights = np.array([w1,w2])

# w = np.array([[[0]*M for i in range(L)] , [[0]*L for i in range(N)]])
# X = X.iloc[:, :]
# Y = Y.iloc[:, 0]

print(w1)  # from x to hidden layer
print(w2)  # from hidden layer to output

learning_rate = 0.03
n_iterations = 500

# w1 = pd.DataFrame(w1)
#
# print(w1.T.shape)
# print(X.shape)

# Ah = f( 𝑾𝒉 ∗ 𝑨𝒉−𝟏 )
print(sum(np.dot(np.array(weights[0]) , np.array(X.T))))

print(np.array(weights[0][1]))

print(np.array(X))
# print(calc_input_to_next_layer(weights,X,2))

