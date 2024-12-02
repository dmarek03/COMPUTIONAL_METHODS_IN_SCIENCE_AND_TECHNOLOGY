import numpy as np
from typing import Final


def generate_Legendre_polynomial(min_range: int, max_range: int):
    p = [np.poly1d([1]), np.poly1d([1, 0])]
    x = np.poly1d([1, 0])
    for i in range(min_range, max_range):
        p.append((2*i+1)/(i+1) * p[i]*x - (i/(i+1))*p[i-1])

    return p


def main() -> None:
    MIN_RANGE: Final = 1
    MAX_RANGE: Final = 6
    polynomials = generate_Legendre_polynomial(MIN_RANGE, MAX_RANGE)
    for poly in polynomials:
        print(poly)


if __name__ == '__main__':
    main()
