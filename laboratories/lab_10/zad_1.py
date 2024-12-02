from numpy import  e, sin, cos
from typing import Callable


def fun(x, y):
    return sin(x)*cos(x) -y*cos(x)

def exact_result(x):
    return e**(-sin(x)) + sin(x) -1


def euler_method(x0:float, y0:float, h:float, f:Callable, interation_num:int)-> tuple[float, float]:
    x_curr = x0
    y_curr = y0
    for _ in range(interation_num):
        k1 = h * f(x_curr, y_curr)
        x_curr += h
        y_curr += k1
    return x_curr, y_curr


def runge_kutta_method(x0:float, y0:float, h: float, f:Callable,interation_num: int):
    x_curr = x0
    y_curr =  y0
    for _ in range(interation_num):

        k1 = h*f(x_curr, y_curr)
        k2 = h*f(x_curr+h/2, y_curr+k1/2)
        k3 = h*f(x_curr+h/2, y_curr+k2/2)
        k4 = h*f(x_curr+h, y_curr+k3)


        x_curr += h
        y_curr += (k1+ 2*k2 + 2*k3 + k4)/6

    return x_curr, y_curr




def main() -> None:
    interation_number = [100, 1000, 10000, 100000, 1000000]


    for i  in [1, 2]:
        for n in interation_number:
            x, y = runge_kutta_method(0, 0, i/n, fun, n)
            exact_res = exact_result(x)
            print(f'Runge Kutta method:')
            print(f'{i/n=}')
            print(f'{n=}')
            print(f'{x=}')
            print(f'{y=}')
            print(f'{exact_res=}')
            print(f'{abs(exact_res-y)=}')
            x1, y1 = euler_method(0, 0, i/n, fun, n)
            exact_res1 = exact_result(x1)
            print(f'Euler method:')
            print(f'{i/n=}')
            print(f'{n=}')
            print(f'{y1=}')
            print(f'{exact_res=}')
            print(f'{abs(exact_res1 - y1)=}')



if __name__ == '__main__':
    main()