import numpy as np


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


def iteration_test(precision):
    A = np.array(
        [[10, -1, 2, -3],
        [1, 10, -1, 2],
         [2, 3, 20, -4],
        [3, 2, 1, 20]]
    )
    b = np.array([0,5,-10, 15]).reshape(-1, 1)
    solves, res = jacobi_iteration_method(A, b, precision)
    print(f'{res=}')


def main() -> None:
    iteration_test(10e-4)
    iteration_test(10e-5)
    iteration_test(10e-6)


if __name__ == '__main__':
    main()
