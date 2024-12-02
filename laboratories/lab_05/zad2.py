import numpy
import  numpy as np
import scipy.integrate as spi
from math import pi

def integrate(fun, r_min:float, r_max: float) -> float:
    return spi.quad(fun, r_min, r_max)[0]


def chebyshev_polynomials(n):
    p = [np.poly1d([1]), np.poly1d([1, 0])]
    x = np.poly1d([1, 0])
    for i in range(2, n+1):
        p.append(2*x*p[i-1] - p[i-2])
    return p

def f(x):
    return 1/(1+x**2)


def approx_fun(n: int):
    CH = chebyshev_polynomials(n)
    c = []
    fA = [np.poly1d([0])]
    for i in range(n+1):
        c.append(
            (1/pi if i == 0 else 2/pi)*spi.quad(
                (lambda x: f(x) * CH[i](x)/(1-x**2)**0.5), -1, 1
            )[0]
        )
        fA.append(c[i] * CH[i])
    return sum(fA)


def integration(n):
    integer = approx_fun(n).integ()
    return integer(1) - integer(-1)


def main() -> None:
    print(f'{integration(8)=}')
    print(f'{abs(integration(8) -pi/2) =}')



if __name__ == '__main__':
    main()


