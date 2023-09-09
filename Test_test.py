import numpy as np
import numdifftools as nd
from datetime import datetime

import numpy as np

start_time = datetime.now()

#тест gd с дихотомией при уменьшение альфы на +- eps (крайне медленно)
def grad_f(func, X: np.array) -> np.array:
    return np.array(nd.Gradient(func)(X))

def fun(x):
    return 7 * (x[0] + 3) ** 2 + 0.5 * (x[1] + 7) ** 2

def dichotomy(f, X, eps=1e-5):
    grad = grad_f(f, X)
    a, b = 0,1
    function_calls = 0
    while abs(a - b) / 2 > eps:
        alpha1 = (a + b - eps) / 2
        alpha2 = (a + b + eps) / 2
        X1, X2 = X - alpha1 * grad, X - alpha2 * grad
        if f(X1) < f(X2):
            b = alpha2
        else:
            a = alpha1
        function_calls += 2
    return (a + b) / 2,function_calls

def dichotomy_grad(f, x, epoch = 20):
    history = [np.hstack((x, f(x)))]
    for _ in range(epoch):
        lr, _ = dichotomy(f, x)
        x = x - lr * grad_f(f, x)
        history.append(np.hstack((x, f(x))))
    return np.array(history)

x = np.array([-20, 10])
points = dichotomy_grad(fun, x)
print(points)
end = datetime.now() - start_time
print(end)