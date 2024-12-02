from numpy import cos, sin, e
from typing import Callable


def f1(x:float) -> float:
    return x*cos(x)-1


def f2(x:float) -> float:
    return x**3 - 5*x - 6


def f3(x:float) -> float:
    return e**(-x) -x**2 + 1


def df1(x:float) -> float:
    return cos(x) - x*sin(x)


def df2(x:float) -> float:
    return 3*x**2 - 5


def df3(x:float) -> float:
    return -e**(-x) - 2*x


def recursive_newton_method(fun: Callable, df: Callable,  x0: int, n: int) -> None:
    root_approximation = x0

    for i in range(n):
        root_approximation -= fun(root_approximation)/df(root_approximation)
        print(f'Approximation number {i}: {root_approximation = }')


def main() -> None:
    print("---------------------------Function_1---------------------------")
    recursive_newton_method(f1, df1, 4, 10)
    print("---------------------------Function_2---------------------------")
    recursive_newton_method(f2, df2, 2, 10)
    print("---------------------------Function_3---------------------------")
    recursive_newton_method(f3, df3, 0, 10)


if __name__ == '__main__':
    main()
