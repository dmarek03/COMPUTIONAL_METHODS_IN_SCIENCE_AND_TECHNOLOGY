from numpy import cos, e, pi, sqrt
from scipy.special import roots_hermite
import timeit

def fun(x:float)->float:
    return cos(x)


def gauss_hermite_approximation(n: int) -> float:

    roots, weights = roots_hermite(n)
    f_r = [fun(r) for r in roots]

    return sum(f_r[i]*weights[i] for i in range(len(roots)))



def calculate_integral_approximation(eps: float, exact_res: float) -> tuple[float, int]:
    n = 0
    integral_approximation = 0
    while abs(exact_res-integral_approximation) >= eps:
        n += 1
        integral_approximation = gauss_hermite_approximation(n)

    return integral_approximation, n


def get_evaluation_time(function_name: str, n: int) -> float:
    return timeit.timeit(stmt=function_name, globals=globals(), number=n)/n


def main() -> None:
    eps = 1e-16
    exact_res = sqrt(pi) / e ** 0.25
    num = 100
    integral_approximation, n =  calculate_integral_approximation(eps, exact_res)
    fun_name_to_evaluation = "calculate_integral_approximation(1e-16, sqrt(pi) / e ** 0.25)"
    gauss_hermite_time = get_evaluation_time(fun_name_to_evaluation,num)
    print(f'{gauss_hermite_time = }')
    print(f'{integral_approximation = }')
    print(f'{n = }')


if __name__ == '__main__':

    main()