import numpy as np
from matplotlib import pyplot as plt


def fun1(x) -> float:
    return 1 + x**3


def fun2(x) -> float:
    return 1.5*x**2 - 0.6*x + 1.05


def draw_plot() -> None:
    d = np.linspace(0, 1, num=1000, dtype=float)
    plt.plot(d, fun1(d), color='green', label='f(x)=x^3+1')
    plt.plot(d, fun2(d), color='darkblue', label='f(x)=1.5x^2-0.6x+1.05')
    plt.legend()
    plt.show()


def main() -> None:
    draw_plot()


if __name__ == '__main__':
    main()

