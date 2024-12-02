from random import randint
from typing import Final
import numpy as np
from numpy import e

def generate_matrix(n:int):
    M = [[0.0]*n for _ in range(n)]
    V = [randint(-1, 0) for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if j - i == 1:
                M[i][j] = round(1/(j+1), 3)
                M[j][i] = round(1/(j+1), 3)

            if i == j:
                if 0 < i < n-1:
                    M[i][i] = 2
                else:
                    M[i][i] = 1
    return M, V


def jacobi_iteration_method(matrix, vector, precision):
    x = np.zeros(len(matrix[0])).reshape(-1, 1)
    D =  np.diag(matrix).reshape(-1,1)
    L_U = matrix-np.diagflat(np.diag(matrix))
    results = []
    norm_vector = np.linalg.norm(vector)
    norm_one = 2
    norm_two = 2
    i = 1

    while norm_one > precision or norm_two > precision:
        next_x = (vector-L_U@x)/D
        norm_one = np.linalg.norm(abs(x-next_x))
        norm_two = np.linalg.norm(matrix@x-vector)/norm_vector
        results.append((i, norm_one, norm_two))
        x =  next_x
        i += 1
    return x, results


def chebyshev_iteration_method(matrix, vector, precision):
    x_prior = np.zeros(len(matrix[0])).reshape(-1, 1)
    t = []
    results = []
    eigs = np.linalg.eig(matrix)[0]
    p, q = np.min(np.abs(eigs)) , np.max(np.abs(eigs))
    r = vector-matrix@x_prior
    x_posterior = x_prior + 2*r/(p+q)
    r = vector - matrix @ x_posterior
    t.append(1)
    t.append(-(p+q)/(q-p))
    beta = -4/(q-p)
    i = 1
    norm_one = 2
    norm_two = 2
    norm_vector = np.linalg.norm(vector)

    while norm_one > precision or norm_two > precision:
        norm_one = np.linalg.norm(abs(x_posterior-x_prior))
        norm_two = np.linalg.norm(matrix@x_posterior-vector)/norm_vector
        results.append((i, norm_one, norm_two))

        i += 1
        t.append(2*t[1]*t[-1]-t[-2])
        alpha= t[-3]/t[-1]
        old_prior, old_posterior = x_prior, x_posterior
        x_prior = old_posterior
        x_posterior = (1+alpha)*old_posterior-alpha*old_prior + (beta*t[-2]/t[-1])*r
        r=vector-matrix@x_posterior

    return x_posterior, results


def main() -> None:
    N:Final = 5
    matrix, vector = generate_matrix(N)
    A = np.array(matrix)
    x = np.array(vector)
    b = A @ x
    print(f'{x=}')
    solves1, res1 = jacobi_iteration_method(A, b, 10e-6)
    solves2, res2 = chebyshev_iteration_method(A, b, 10e-6)
    print("Matrix A:")
    print(A)
    print("Vector b:")
    print(b)
    print("Solutions vector:")
    print(f'{solves1=}')
    print(f'{solves2=}')
    print("--------------Jacobi_iteration_method--------------")
    for t in res1:
        print(t[0], " ", t[1], " ", t[2])
    print("--------------Chebyshev_iteration_method--------------")
    for t in res2:
        print(t[0], " ", t[1], " ", t[2])


if __name__ == '__main__':
    main()