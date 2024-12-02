from numpy import sin, cos, pi

def exact_result(x):
    return cos(x) - sin(x) + x


def fun(x, y, y_prim):
    return x-y


def runge_kutta_for_hit(num_iteration, h, x0, y0, a, func):
    x_curr = x0
    y_curr = y0
    for _ in range(num_iteration):
        k1 = h * func(x_curr, y_curr, a)
        k2 = h * func(x_curr + h / 2, y_curr + k1 / 2, a)
        k3 = h * func(x_curr + h / 2, y_curr + k2 / 2, a)
        k4 = h * func(x_curr + h, y_curr + k3, a)

        delta_a = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        k1 = h * a
        k2 = h * (a +  h * func(x_curr + h / 2, y_curr + k1 / 2, a))
        k3 = h * (a + h*func(x_curr + h / 2, y_curr + k2 / 2, a))
        k4 = h * (a + h*func(x_curr + h, y_curr + k3, a))

        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x_curr += h
        y_curr += delta_y
        a += delta_a


    return x_curr, y_curr
# y'' = f(x, y, y')
# y(x0) = y0
# y(x1) = y1


def hit_method(final_iterations, a0, a1, x0, y0, x1,y1, func, h, epsilon):
    # Number of iterations to reach x1 from x0 with step h
    bisect_iterations = int((x1-x0)/h)
    a = (a0+a1)/2
    y =  runge_kutta_for_hit(bisect_iterations, h, x0, y0, a, func)[1]
    i = 0

    while abs(y-y1) > epsilon:
        if (y-y1)*(runge_kutta_for_hit(bisect_iterations, h, x0, y0, a0, func)[1] -y1) > 0:
            a0 = a
        else:
            a1 = a

        a = (a0 + a1)/2

        y = runge_kutta_for_hit(bisect_iterations, h, x0, y0, a, func)[1]
        i += 1
    # now we have
    # ya'' = f(x, ya, ya')
    # ya(x0) =  y0
    # ya' =  a

    return runge_kutta_for_hit(final_iterations, h, x0, y0, a, func)



def main()->None:
    for x_val in [0.5, 1, 2]:
        for n in [100, 1000, 10000, 100000, 1000000]:
            x, y = hit_method(n, -100, 100, 0, 1, pi/2, pi/2 -1, fun, x_val/n, 1e-3)
            print(f'{x_val/n =}')
            print(f'{abs(y-exact_result(x))=}')


if __name__ == '__main__':
    main()