from typing import Callable, Final
from math import sqrt, pi
from random import uniform


def fun_1(x:float) -> float:
    return x**2 + x + 1


def fun_2(x:float) -> float:
    return sqrt(1-x**2)


def fun_3(x:float) -> float:
    return 1/sqrt(x)





def hit_and_miss(a:float, b:float, n:int, h:float, fun:Callable[[float], float]):
    S = 0
    for i in range(n):
        x = uniform(a, b)
        y = uniform(0, h)

        if y < fun(x):

            S += 1
    return ((b-a)*h)*S/n



def main() ->None:
    A:Final = 0
    B:Final = 1
    INTEGRAL_1:Final = 11/6
    INTEGRAL_2: Final = pi/4
    INTEGRAL_3: Final = 2

    num_of_interation = [10**2, 10**3,10**4, 10**5, 10**6, 10**7]

    # Maximum values of functions on [0, 1]
    H_1 = 3
    H_2 = 1
    H_3 = 10  # Arbitrary large value since fun_3 is problematic as x approaches 0


    for n in num_of_interation:
        error_1 = round(abs(INTEGRAL_1 - hit_and_miss(A, B, n, H_1, fun_1)), 6)
        error_2 = round(abs(INTEGRAL_2 - hit_and_miss(A, B, n, H_2, fun_2)), 6)
        error_3 = round(abs(INTEGRAL_3 - hit_and_miss(A, B, n, H_3, fun_3)), 6)

        print(f'n={n}, error_1={error_1}, error_2={error_2}, error_3={error_3}')


if __name__ == '__main__':
    main()
