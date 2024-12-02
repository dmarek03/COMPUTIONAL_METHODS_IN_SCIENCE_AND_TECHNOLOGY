
def f1(x:float, y:float) -> float:
    return x**2 + x*y**3 - 9

def f2(x:float, y: float) -> float:
    return 3*x**2*y - y**3 - 4


def df1x(x:float, y:float) -> float:
    return 2*x + y**3


def df1y(x:float, y:float) -> float:
    return 3*x*y**2


def df2x(x:float, y:float) -> float:
    return 6*x*y


def df2y(x:float, y:float) -> float:
    return 3*x**2 - 3*y**2


def Jacob(x:float, y:float) -> float:
    return df1x(x, y) * df2y(x, y) - df1y(x, y) * df2x(x, y)


def Newton(x0:float, y0:float, n: int) -> None:
    x1 = x0
    x2 = y0

    for i in range(1, n+1):
        delta_x1 = (f2(x1, x2)*df1y(x1, x2) - f1(x1, x2)*df2y(x1, x2))/Jacob(x1, x2)
        delta_x2 = (f1(x1, x2) * df2x(x1, x2) - f2(x1, x2) * df1x(x1, x2)) / Jacob(x1, x2)

        x1 += delta_x1
        x2 += delta_x2

        print(f'Iteration number {i}: {x1 = } {x2 = }')



def main() -> None:
    Newton(0, 2, 10)


if __name__ == '__main__':
    main()


