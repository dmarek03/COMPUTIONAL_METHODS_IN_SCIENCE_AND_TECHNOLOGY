from typing import Any
import numpy as np
from numpy import  cos, e
from numpy import ndarray, dtype
from typing import Callable
import timeit

def fun(x:float|ndarray[Any, dtype[Any]]) -> float:
    return e**(-x**2)*cos(x)

def rectangle_method(r_min: float, r_max: float, n: int) -> float:
    width =  (r_max-r_min)/n
    integral_approx = 0
    pieces = np.linspace(r_min, r_max-width, n)
    pieces += width/2
    for x in pieces:
        integral_approx += fun(x)*width
    return integral_approx


def trapezoidal_method(r_min: float, r_max: float, n: int) -> float:
    width = (r_max-r_min)/n
    pieces = np.linspace(r_min, r_max, n+1)
    integral_approx = 0
    for i in range(1, n+1):
        integral_approx += width*(fun(pieces[i]) + fun(pieces[i-1]))/2

    return integral_approx

def simpson_method(r_min:float, r_max:float , n: int) -> float:
    width = (r_max-r_min)/n
    integral_approx = 0
    pieces = np.linspace(r_min, r_max, n+1)
    for i in range(1, n// 2 + 1):
        integral_approx += (fun(pieces[2*i-2]) + 4*fun(pieces[2*i-1]) + fun(pieces[2*i]))
    return (integral_approx*width)/3


def approximate_integral(function: Callable,exact_res: float, eps: float) -> tuple[float, int, int]:
    integral_approx =  0
    r_min = 0
    r_max = 0

    while abs(exact_res-integral_approx) > eps:
        r_min -= 1
        r_max += 1
        n = (r_max-r_min)*10
        integral_approx = function(r_min, r_max, n)

    return integral_approx, r_min, r_max


def get_evaluation_time(function: str, n: int) -> float:
    evaluation_time = timeit.timeit(stmt=function, globals=globals(), number=n)
    return evaluation_time/n

def main() -> None:
     eps = 1e-16
     exact_res = np.sqrt(np.pi)/ np.e**0.25
     n = 100

     print("-------------------Rectangle method-------------------")
     rectangle_method_res, r_min, r_max = approximate_integral(rectangle_method, exact_res, eps)
     rectangle_func_to_evaluation = "approximate_integral(rectangle_method, np.sqrt(np.pi)/ np.e**0.25,1e-16)"
     rectangle_evaluation_time = get_evaluation_time(rectangle_func_to_evaluation,n)
     print(f'{rectangle_evaluation_time = }')
     print(f'{rectangle_method_res = }')
     print(f'{abs(exact_res-rectangle_method_res) = }')
     print(f'{r_min = }')
     print(f'{r_max = }')
     print("-------------------Trapezoidal method-------------------")
     trapezoidal_method_res, r_min, r_max = approximate_integral(trapezoidal_method, exact_res, eps)
     trapezoidal_func_to_evaluation = "approximate_integral(trapezoidal_method,np.sqrt(np.pi)/ np.e**0.25,1e-16)"
     trapezoidal_evaluation_time = get_evaluation_time(trapezoidal_func_to_evaluation, n)
     print(f'{trapezoidal_evaluation_time = }')
     print(f'{trapezoidal_method_res = }')
     print(f'{abs(exact_res-trapezoidal_method_res) = }')
     print(f'{r_min = }')
     print(f'{r_max = }')
     print("-------------------Simpson's method-------------------")
     simpson_method_res,r_min, r_max = approximate_integral(simpson_method, exact_res, eps)
     simpson_func_to_evaluation = "approximate_integral(simpson_method,np.sqrt(np.pi)/ np.e**0.25,1e-16)"
     simpson_evaluation_time = get_evaluation_time(simpson_func_to_evaluation, n)
     print(f'{simpson_evaluation_time = }')
     print(f'{simpson_method_res = }')
     print(f'{abs(exact_res-simpson_method_res) = }')
     print(f'{r_min = }')
     print(f'{r_max = }')

if __name__ == '__main__':
    main()
