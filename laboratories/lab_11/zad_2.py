from time import time
import numpy as np
from typing import Callable, Final
from math import sqrt, pi,inf
from random import uniform
from matplotlib import pyplot as plt


def fun_1(x:float) -> float:
    return x**2 + x + 1


def fun_2(x:float) -> float:
    return sqrt(1-x**2)


def fun_3(x:float) -> float:
    return 1/sqrt(x)


def hit_and_miss_approximate(a:float, b:float, h:float, eps:float,exact_value:float, fun:Callable[[float], float]):
    S = 0
    P = (b-a)*h
    n = 1
    while abs(exact_value -P*S/n) > eps:
        x = uniform(a, b)
        y = uniform(0, h)

        if y < fun(x):

            S += 1
        n += 1


    return P*S/n


def rectangle_approximate(r_min:float, r_max:float, eps:float, exact_res:float,fun:Callable):
    n = 1
    integral_approx = 0
    while not abs(integral_approx - exact_res) <= eps:
        width = (r_max - r_min) / n
        pieces = np.linspace(r_min, r_max - width, n)
        pieces += width / 2

        integral_approx = sum(fun(x) for x in pieces)*width
        n  += 1
    return integral_approx


def get_times(r_min:float, r_max:float, h:int,accuracies,exact_res,function):
    times = [[0.0]*len(accuracies) for _ in range(2)]
    res = [[0.0]*len(accuracies) for _ in range(2)]

    for eps in accuracies:
        start_time = time()
        res[0][accuracies.index(eps)] = rectangle_approximate(r_min, r_max, eps, exact_res, function)
        end_time = time()
        times[0][accuracies.index(eps)] = round(end_time-start_time, 6)
        start_time_1 = time()
        res[1][accuracies.index(eps)] = hit_and_miss_approximate(r_min, r_max,h, eps, exact_res, function)
        end_time_1 = time()
        times[1][accuracies.index(eps)] = round(end_time_1 - start_time_1, 6)

    return res, times


def plot_times(accuracies, times_rec, times_mc, title):
    plt.figure()
    plt.plot(times_rec,accuracies, label="Rectangle Method", marker='o')
    plt.plot(times_mc, accuracies, label="Hit-and-Miss Method", marker='o')
    plt.yscale('log')
    plt.ylabel('Accuracy (eps)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def main() -> None:
    A: Final = 0
    B: Final = 1
    INTEGRAL_1: Final = 11 / 6
    INTEGRAL_2: Final = pi / 4
    INTEGRAL_3: Final = 2


    # Maximum values of functions on [0, 1]
    H_1 = 3
    H_2 = 1
    H_3 = 10  # Arbitrary large value since fun_3 is problematic as x approaches 0

    accuracies = [1e-3, 1e-4, 1e-5, 1e-6]


    res_1, times_1 =get_times(A, B, H_1, accuracies, INTEGRAL_1, fun_1)
    res_2, times_2 = get_times(A, B, H_2, accuracies, INTEGRAL_2, fun_2)
    res_3,times_3 =  get_times(A, B, H_3, accuracies, INTEGRAL_3, fun_3)


    plot_times(accuracies,times_1[0], times_1[1], "Methods time efficiency for f(x) = x**2 + x + 1")
    plot_times(accuracies, times_2[0], times_2[1], "Methods time efficiency for f(x) = sqrt(1-x**2)")
    plot_times(accuracies, times_3[0], times_3[1], "Methods time efficiency for f(x) = 1/sqrt(x)")



if __name__ == '__main__':
    main()