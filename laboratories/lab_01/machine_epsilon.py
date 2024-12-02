def find_machine_epsilon(basis: int, precision: int) -> float:
    return basis**(1-precision)


print(f'Machine epsilon for binary system w precision equal 11  = {find_machine_epsilon(2, 11)}')
