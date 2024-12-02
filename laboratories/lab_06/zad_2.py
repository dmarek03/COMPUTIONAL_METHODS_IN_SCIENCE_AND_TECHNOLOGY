import timeit
from numpy import e, cos,sqrt, pi,  ndarray, dtype
from typing import Any

def fun(x:float|ndarray[Any, dtype[Any]]) -> float:
    return e**(-x**2)*cos(x)


def quad_simpsons_mem(f, a, b):
    m = (a+b)/2
    return m, abs(b-a)/6 *(f(a) +4*f(m) + f(b))


def _quad_asr(f, a, b, eps, whole, m):
    lm, left = quad_simpsons_mem(f, a, m)
    rm,right = quad_simpsons_mem(f, m, b)

    delta = left + right - whole
    if abs(delta) <= 15*eps:
        return  left + right + delta/15

    return _quad_asr(f, a, m, eps/2, left, lm) + _quad_asr(f, m, b, eps/2, right, rm)



def quad_asr(f, a, b, eps):
    
    m,whole = quad_simpsons_mem(f, a, b)
    return  _quad_asr(f, a, b, eps, whole, m)


def calculate_integral_adaptive_method(exact_res: float, eps: float) -> tuple[float, int, int]:
    r_min = 0
    r_max = 0
    integral_approximation = 0
    while abs(exact_res - integral_approximation) > eps:
        r_min -= 1
        r_max += 1
        integral_approximation = quad_asr(fun, r_min, r_max, eps)

    return integral_approximation, r_min, r_max



def get_evaluation_time(function: str, n: int) -> float:
    evaluation_time = timeit.timeit(stmt=function, globals=globals(), number=n)
    return evaluation_time/n



def main() -> None:
    eps = 1e-16
    exact_res = sqrt(pi)/e**0.25
    n = 10
    adaptive_integral, r_min, r_max = calculate_integral_adaptive_method(exact_res, eps)
    adaptive_func_to_evaluation = "calculate_integral_adaptive_method(sqrt(pi)/e**0.25,1e-16)"
    adaptive_method_time = get_evaluation_time(adaptive_func_to_evaluation,n)
    print(f'{adaptive_method_time = }')
    print(f'{adaptive_integral = }')
    print(f'{abs(exact_res-adaptive_integral) = }')
    print(f'{r_min = }')
    print(f'{r_max = }')


if __name__ == '__main__':
    main()


