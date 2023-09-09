import numpy as np

phi = (np.sqrt(5) + 1) / 2

def F(x):
    return x ** 3 / 3 - x ** 2 / 2 - x - 1

def golden_section_search(func, a, b, eps):
    c = b - (b - a) / phi
    d = a + (b - a) / phi
    Fc = F(c)
    Fd = F(d)
    while abs(b - a) > eps:
        if Fc < Fd:
            b = d
            d = c
            Fd = Fc
            c = b + (a - b) / phi
            Fc = F(c)
        else:
            a = c
            c = d
            Fc = Fd
            d = a + (b - a) / phi
            Fd = F(d)
    return  (a + b) / 2

print(golden_section_search(F,1, 2, 0.000001))