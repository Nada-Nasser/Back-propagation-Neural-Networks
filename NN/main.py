import matplotlib.pyplot as plt
import pandas as pd

from helpers import *

line = np.array(pd.read_csv("train.txt", header=None, delim_whitespace=True, nrows=2))
data = pd.read_csv("train.txt", skiprows=[0, 1], header=None, delim_whitespace=True)

M, L, N, K = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[1][0])

X_data = (data.iloc[:,0:M])
X_data = (X_data - X_data.mean())/ X_data.std()

X = np.array(X_data)
Y = np.array(data.iloc[:, M:])


print("M = ", M)  # N is number of Input Nodes
print("L = ", L)  # L is number of Hidden Nodes
print("N = ", N)  # N is number of Output Nodes
print("K = ", K)  # K, the number of training examples
# each line has length M+N values,
# first M values are X vector
# last N values are output values.

n, m = X.shape
X0 = np.ones((n, 1))
X = np.hstack((X0, X))
M += 1

w1 = []
for l in range(L):
    w1.append(np.array(np.random.randn(M)*0.00001))

w2 = []
for n in range(N):
    w2.append(np.array(np.random.randn(L)*0.00001))


print(w2)

alpha = 0.0003
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
