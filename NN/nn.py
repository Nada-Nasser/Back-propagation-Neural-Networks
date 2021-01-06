import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from helpers import *


NHL = 1
Ls = []

line = np.array(pd.read_csv("d.txt", header=None, delim_whitespace=True, nrows=2))
M, L, N, K = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[1][0])
X = np.array(pd.read_csv("d.txt", skiprows=[0, 1], header=None, delim_whitespace=True , usecols = [i for i in range(M)]))
Y = np.array(pd.read_csv("d.txt", skiprows=[0, 1], header=None, delim_whitespace=True).iloc[:, M:])

print(X)
print("M = ", M)  # N is number of Input Nodes
print("L = ", L)  # L is number of Hidden Nodes
print("N = ", N)  # N is number of Output Nodes
print("K = ", K)  # K, the number of training examples
# each line has length M+N values,
# first M values are X vector
# last N values are output values.

# for i in range(K):
#     X[i] = (X[i] - np.mean(X[i]))/ np.std(X[i])
#
#
# print(X)

# n,m = X.shape
# X0 = np.ones((n,1))
# X = np.hstack((X0,X))
# M+=1

Ls.append(M)
Ls.append(L)
Ls.append(N)
weights = []

print(Ls)

for i in range(0,len(Ls)-1):
    # print(i)
    w = np.array([[0]*Ls[i] for j in range(Ls[i+1])])
    weights.append(w)

# w1 = np.array([[0]*M for i in range(L)])
# w2 = np.array([[0]*L for i in range(N)])
# weights = np.array([[],w1,w2])
#
weights = [[[0.3 , -0.9 , 1 ],[-1.2 , 1 ,1 ]] , [1 , 0.8]]
print(weights)



# Ah = f( 𝑾𝒉 ∗ 𝑨𝒉−𝟏 )

# print(weights[0])
#
learning_rate = 0.03
n_iterations = 500

print(X)

(gradient_descent(X,Y,M,NHL,Ls,N,K,weights,learning_rate , 1))



#H = n_H_layers
#L1 L2 L3... LH