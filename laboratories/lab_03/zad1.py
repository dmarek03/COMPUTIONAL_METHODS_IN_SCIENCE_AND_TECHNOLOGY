import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inter
from typing import Final


def Runge_function(x) -> float:
    return 1/(1 + 25*x**2)


def draw_plot(min_range: int, max_range: int) -> None:
    d = np.linspace(-1, 1, num=1000, dtype=float)
    for n in range(min_range, max_range):
        knots = np.linspace(-1, 1, num=n, dtype=float)
        f_int = inter.KroghInterpolator(knots, Runge_function(knots))
        plt.plot(d, f_int(d), color='green')
        font = {
            'family': 'cambria',
            'color': 'black',
            'weight': 'bold',
            'style': 'italic',
            'size': 10
        }
        plt.title(
            f"Rysunek {n-min_range+1}: "
            f"Porównanie funkcji interpolującej oraz interpolowanej dla n = {n-2}",
            fontdict=font,
            fontsize=font['size']
        )
        plt.plot(d, Runge_function(d), color='red')
        plt.show()


def main() -> None:
    MIN_RANGE: Final = 4
    MAX_RANGE: Final = 14
    draw_plot(MIN_RANGE, MAX_RANGE)


if __name__ == '__main__':
    main()

