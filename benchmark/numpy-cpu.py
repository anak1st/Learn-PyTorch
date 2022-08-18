import time
import numpy as np

def numpy_cpu(size, flag=False):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    start = time.perf_counter()
    c = a @ b
    end = time.perf_counter()
    cost = end - start
    if flag:
        print(f"CPU Elapsed time is {cost:.5f} s")
    return cost

if __name__ == "__main__":
    size = 7500
    cost_cpu = 0
    times = 5
    for i in range(0, times):
        cost_cpu += numpy_cpu(size, True)
    cost_cpu /= times
    print(f"CPU Elapsed Average time is {cost_cpu:.5f} s")
