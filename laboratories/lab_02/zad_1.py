from math import e

def find_machine_epsilon(basis: int, precision: int) -> float:
    return basis**(1-precision)


def find_exp(x: int, epsilon: float) -> float:
    exponents = 0
    element = 1
    i = 1
    is_negative_index = False
    if x < 0:
        is_negative_index = True
        x = abs(x)
    while element > epsilon:
        exponents += element
        element *= x / i
        i += 1

    return 1/exponents if is_negative_index else exponents


if __name__ == '__main__':
    machine_epsilon = find_machine_epsilon(2, 53)
    args = [-1, 1, -5, 5, -10, 10]
    print(f'{machine_epsilon=}')
    for a in args:
        #print(find_exp(a, machine_epsilon))
        # print(f'{e**a}')
        print(abs(find_exp(a, machine_epsilon) - e**a)/e**a)




