from typing import Any, Final
import numpy as np
from math import pi
from numpy import ndarray, dtype

def fun(x:float|ndarray[Any, dtype[Any]]) -> float:
    return 1/(x**2+1)

def rectangle_method(r_min: int, r_max: int, n: int) -> float:
    width =  (r_max-r_min)/n
    integral_approx = 0
    pieces = np.linspace(r_min, r_max-width,n)
    pieces += width/2
    for x in pieces:
        integral_approx += fun(x)*width

    return integral_approx


def trapezoidal_method(r_min:int, r_max: int, n: int) -> float:
    pieces = np.linspace(r_min, r_max, n+1)
    return np.trapz(fun(pieces), pieces)



def simpson_method(r_min: int, r_max: int, n: int) -> float:
    width = (r_max-r_min)/n
    integral_approx = 0
    pieces = np.linspace(r_min, r_max, n+1)
    for i in range(1, n// 2 + 1):
        integral_approx += (fun(pieces[2*i-2]) + 4*fun(pieces[2*i-1]) + fun(pieces[2*i]))
    return (integral_approx*width)/3




def main() -> None:
     MIN_RANGE: Final = 0
     MAX_RANGE: Final = 1
     n = 5
     real_integral_value =  pi/4
     print(f'{real_integral_value=}')
     print("Rectangle method")
     rectangle_method_res = rectangle_method(MIN_RANGE, MAX_RANGE, n)
     print(rectangle_method_res)
     print(abs((rectangle_method_res-real_integral_value)/rectangle_method_res))
     print("Trapezoidal method")
     trapezoidal_method_res = trapezoidal_method(MIN_RANGE, MAX_RANGE, n)
     print(trapezoidal_method_res)
     print(abs((trapezoidal_method_res - real_integral_value) / trapezoidal_method_res))
     print("Simpson's method")
     simpson_method_res = simpson_method(MIN_RANGE, MAX_RANGE, n)
     print(simpson_method_res)
     print(abs((simpson_method_res - real_integral_value) / simpson_method_res))

if __name__ == '__main__':
    main()
