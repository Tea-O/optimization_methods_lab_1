import math

import numpy as np


def fibonacci(n):
    if n <= 0:
        return [0]
    elif n == 1:
        return [0, 1]
    else:
        fib_sequence = [0, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence


def fib(n):
    x1 = 1; x2 = 1
    for i in range(3, n+1):
        t = x1 + x2
        x1 = x2
        x2 = t
    return x2
def fibonacci_search(func, a, b, epsilon):
     n = 1
     while (b - a) / fib(n) >= epsilon:
         n+=1
     c = a + (b - a) * fib(n - 2)/fib(n)
     d = a + (b - a) * fib(n - 1)/fib(n)
     while n >= 2:
        n-=1
        if func(c) <= func(d):
            b = d
            d = c
            c = a + (b - a) * fib(n - 2)/fib(n)
        else:
            a = c
            c = d
            d = a + (b - a) * fib(n - 1)/fib(n)
     return (b + a) / 2
def F(x):
    return x ** 3 / 3 - x ** 2 / 2 - x - 1

print(fibonacci_search(F,1, 2, 0.000001))