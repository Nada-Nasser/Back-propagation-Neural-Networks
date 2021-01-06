import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import *


line = np.array(pd.read_csv("train.txt", header=None, delim_whitespace=True, nrows=2))
data = pd.read_csv("train.txt", skiprows=[0, 1], header=None, delim_whitespace=True)

M, L, N, K = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[1][0])
X = np.array((data.iloc[:,0:M]))
Y = np.array(data.iloc[:, M:])


print("M = ", M)  # N is number of Input Nodes
print("L = ", L)  # L is number of Hidden Nodes
print("N = ", N)  # N is number of Output Nodes
print("K = ", K)  # K, the number of training examples
# each line has length M+N values,
# first M values are X vector
# last N values are output values.

for i in range(K):
    X[i] = (X[i] - np.mean(X[i]))/ np.std(X[i])

n, m = X.shape
X0 = np.ones((n, 1))
X = np.hstack((X0, X))
M += 1

w1 = []
for l in range(L):
    w1.append(np.array(np.random.randn(M)*0.1))

w2 = []
for n in range(N):
    w2.append(np.array(np.random.randn(L)*0.1))

alpha = 0.03
n_iterations = 100
w1, w2, cost = gradient_descent(X, Y, M, L, N, K, w1, w2, alpha, n_iterations)

print(cost)
# draw error graph
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.arange(n_iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('total cost graph')
plt.show()
