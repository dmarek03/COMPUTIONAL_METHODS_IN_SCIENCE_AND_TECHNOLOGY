import math


def calculate_forward_error(exact_fun, approximate_fun) -> float:
    return abs(exact_fun-approximate_fun)


def calculate_backward_error(exact_value, approximate_value) -> float:
    return abs(approximate_value-exact_value)


def sin(x: float) -> float:
    return math.sin(x)


def approximate_fun(x: float, exact: int) -> float:
    return x if exact == 1 else x - (x**3)/6


def asin(x: float) -> float:
    return math.asin(x)


def find_forward_backward_error(args: list[float], approximate_exact, exact_fun,  approximate_fun, approximate_value):
    for a in args:
        print(f"approximate_value= {round(approximate_value(a), 10)}")
        print(f'Forward error for value = {a} : {round(calculate_forward_error(exact_fun(a), approximate_fun(a, approximate_exact)),10)}')
        print(f'Backward error for value = {a} : {round(calculate_backward_error(exact_fun(a), approximate_value(approximate_fun(a, approximate_exact))),10)}')


if __name__ == '__main__':
    find_forward_backward_error([0.1, 0.5, 1.0], 1, sin, approximate_fun, asin)
    find_forward_backward_error([0.1, 0.5, 1.0], 2, sin, approximate_fun, asin)