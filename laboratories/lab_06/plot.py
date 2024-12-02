from matplotlib import  pyplot as plt
import numpy as np
from numpy import e, cos
from typing import Final

def fun(x) -> float:
    return e**(-x**2)*cos(x)


def draw_plot(r_min: int,r_max: float, n: int) -> None:
    d = np.linspace(r_min, r_max, n)
    plt.plot(d, fun(d), color='green')
    font = {
        'family': 'cambria',
        'color': 'black',
        'weight': 'bold',
        'style': 'italic',
        'size': 10
    }
    plt.title(
        f"Wykres funkcji podcałkowej",
        fontdict=font,
        fontsize=font['size']
    )
    plt.plot(d, fun(d), color='red')
    plt.show()


def main()->None:
    R_MIN: Final = -10
    R_MAX: Final = 10
    n = 10000
    draw_plot(R_MIN, R_MAX, n)


if __name__ == '__main__':
    main()