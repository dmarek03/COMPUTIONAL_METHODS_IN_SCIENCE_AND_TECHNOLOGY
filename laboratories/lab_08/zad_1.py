import time
from random import randint
import numpy as np
from numpy import e
from copy import deepcopy
from functools import wraps
from typing import Final
from matplotlib import pyplot as plt


def get_time_evaluation(func):
    @wraps(func)
    def get_time_evaluation_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args,**kwargs)
        end_time = time.perf_counter()
        evaluation_time = end_time-start_time
        print(f'Function {func.__name__}() finished in: {evaluation_time:.6f} secs')
        return value

    return get_time_evaluation_wrapper


def get_time(func,*args) -> float:
    start_time = time.perf_counter()
    func(*args)
    end_time = time.perf_counter()
    evaluation_time = end_time-start_time
    return evaluation_time


def get_det(matrix):
    tmp_matrix = deepcopy(matrix)
    n = len(tmp_matrix)
    p = 1
    sign = 1

    for k in range(n):
        if tmp_matrix[k][k] == 0:
            i = k+1
            while i < n and tmp_matrix[i][k] == 0:
                i += 1

            if i == n:
                return 0
            else:
                tmp_matrix[k], tmp_matrix[i] = tmp_matrix[i], tmp_matrix[k]
                sign = -sign


        for i in range(k+1, n):
            for j in range(k+1, n):
                tmp_matrix[i][j] = (tmp_matrix[k][ k] * tmp_matrix[i][j] - tmp_matrix[i][k] * tmp_matrix[k][j]) / p

        p = tmp_matrix[k][k]


    return sign* tmp_matrix[n - 1][n - 1]


def find_inverse_matrix(matrix):

    if get_det(matrix) == 0:
        raise ValueError("Cannot inverse a singular matrix")
    n = len(matrix)
    tmp_matrix = deepcopy(matrix)
    helper = [[0 if i != j else 1 for i in range(n)] for j in range(n)]

    for i in range(n - 1, 0, -1):
        if tmp_matrix[i][0] > tmp_matrix[i - 1][0]:
            tmp_matrix[i], tmp_matrix[i - 1] = tmp_matrix[i - 1], tmp_matrix[i]
            helper[i], helper[i - 1] = helper[i - 1], helper[i]

    for i in range(n):
        for j in range(n):
            if i != j:
                factor = tmp_matrix[j][i] / tmp_matrix[i][i]

                for k in range(n):
                    tmp_matrix[j][k] -= factor * tmp_matrix[i][k]
                    helper[j][k] -= factor * helper[i][k]

    for i in range(n):
        factor = tmp_matrix[i][i]
        for j in range(n):
            tmp_matrix[i][j] /= factor
            helper[i][j] /= factor


    return helper


def multipy_matrix(A, B):
    if len(A[0]) != len(B):
        raise ValueError("The number of column in first matrix is not equal to the number of rows in second matrix")
    if isinstance(B, list):
        if all(isinstance(elem, list) for elem in B):
            return [[sum(A[i][k] * B[k][j] for k in range(len(A[0]))) for j in range(len(B))] for i in range(len(A))]

        return [sum(A[i][j]*B[j] for j in range(len(B))) for i in range(len(A))]


    return [sum(A[i][j] * B[j] for j in range(len(B))) for i in range(len(A))]


@get_time_evaluation
def solve_using_inverse_matrix(matrix, B):
    return [round(x,8) for x  in multipy_matrix(find_inverse_matrix(matrix), B)]


def LU(matrix):
    n = len(matrix)
    L = [[0.0]*n for _ in range(n)]
    U = [[0 if i !=j else 1 for i in range(n)] for j in range(n)]

    for j in range(n):

        for i in range(j, n):
            res = 0
            for k in range(j):
                res += L[i][k]*U[k][j]
            L[i][j] = matrix[i][j] - res


        for i in range(j+1,n):
            res = 0
            for k in range(j):
                res += L[j][k]*U[k][i]
            if L[j][j] == 0:
                L[j][j] = e-40
            U[j][i] = (matrix[j][i]-res)/L[j][j]

    return np.array(L), np.array(U)


def forward_sub(L, B):
    n = len(B)
    y = [0]*n


    for i in range(n):
        tmp = B[i]
        for j in range(i):
            tmp -= L[i, j]*y[j]
        y[i] = tmp/L[i, i]
    return y


def back_sub(U, B):
    n = len(B)
    x = [0]*n

    for i in range(n-1, -1, -1):
        tmp = B[i]
        for j in range(i+1, n):
            tmp -= U[i, j]*x[j]
        x[i] = tmp/U[i, i]
    return x


@get_time_evaluation
def solve_using_LU(matrix,B):
    L, U =  LU(matrix)
    y =  forward_sub(L,B)
    return [round(xi, 8) for xi in back_sub(U,y)]


def qr(matrix):

    orthonormal_set = gram_schmidt(matrix)
    Q = create_Q(orthonormal_set, matrix)
    R = create_R(orthonormal_set, matrix)
    return Q, R


def create_R(orthonormal_set, matrix):


    R = np.zeros((matrix.shape[1], matrix.shape[1]))
    for i in range(matrix.shape[1]):
        for j in range(i, matrix.shape[1]):
            if i == j:
                R[i, i] = np.linalg.norm(orthonormal_set[i])
            else:
                R[i, j] = np.inner(matrix[:, j], orthonormal_set[i] / np.linalg.norm(orthonormal_set[i]))
    return R


def create_Q(orthonormal_set, matrix):

    Q = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(len(orthonormal_set)):
        Q[:, i] = orthonormal_set[i] / np.linalg.norm(orthonormal_set[i])
    return Q


def gram_schmidt(matrix):

    orthonormal_set = [matrix[:, 0]]
    for i in range(1, matrix.shape[1]):
        factor = 0
        for j in range(i):
            factor += (np.inner(matrix[:, i], orthonormal_set[j]) / np.inner(orthonormal_set[j], orthonormal_set[j])) * (orthonormal_set[j])
        orthonormal_set.append(matrix[:, i] - factor)
    return orthonormal_set


def transpose_matrix(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix))]


@get_time_evaluation
def solve_using_qr(matrix, B):
    q,r = qr(np.array(matrix))
    transposed_q = transpose_matrix(q)
    inverse_r = find_inverse_matrix(r)
    tmp = multipy_matrix(inverse_r, transposed_q)
    return [round(xi, 8) for xi  in multipy_matrix(tmp, B)]


def generate_matrix(size: int):
    return [[randint(0, 99) for _ in range(size)] for _ in range(size)]


def generate_vector(size:int):
    return [randint(0, 99) for _ in range(size)]


def check_solution_correctness(matrix, b, solution:list[float]) -> bool:
    correct_solutions = [round(x, 8) for x in np.linalg.solve(matrix, b)]
    for i in range(len(correct_solutions)):
        if abs(correct_solutions[i]-solution[i]) > e**(-12):
            return False
    return True


def get_data_for_plot(r_min:int, r_max:int, n:int) -> tuple[list[int], list[float]]:
    x_step = (r_max-r_min)//n
    x_data = [r_min + x_step*i for i in range(n+1)]
    y_data = [round(get_time(solve, x), 6) for x in x_data]

    return x_data, y_data


def draw_plot(r_min: int,r_max: int, n: int) -> None:

    x_data,y_data = get_data_for_plot(r_min, r_max, n)
    print(f'{x_data=}')
    print(f'{y_data=}')
    plt.plot(x_data, y_data, color='blue')
    font = {
        'family': 'cambria',
        'color': 'black',
        'weight': 'bold',
        'style': 'italic',
        'size': 12
    }
    plt.title(
        f"Czas rozwiązania układu równań wzgledem liczby równań",
        fontdict=font,
        fontsize=font['size']
    )
    plt.show()



def solve(n):
    matrix = generate_matrix(n)
    b = generate_vector(n)

    x1 = solve_using_inverse_matrix(matrix, b)
    print(f'{x1=}\n')
    print(f'First method solution is correct: {check_solution_correctness(matrix, b, x1)}\n')

    x2 = solve_using_LU(matrix, b)
    print(f'{x2=}\n')
    print(f'Second method solution is correct: {check_solution_correctness(matrix, b, x2)}\n')

    x3 = solve_using_qr(matrix, b)
    print(f'{x3=}\n')
    print(f'Third method solution is correct: {check_solution_correctness(matrix, b, x3)}\n')

def main() -> None:
    N:Final = 10
    #solve(N)
    total_time = get_time(solve, N)
    print(f'Total evaluation time: {total_time:.6f} secs\n')
    draw_plot(10, 100, 5)


if __name__ == '__main__':
    main()
