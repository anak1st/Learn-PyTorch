import math
import numpy as np

def sin(x):
    y = 0
    f = 1
    for i in range(1, 1000, 2):
        t = x**i / math.factorial(i)
        print(t)
        if t < 1e-5:
            break
        y += f * t
        f *= -1

    return y


x = int(input())
y = sin(x)
print("{:.8f}".format(y))

    