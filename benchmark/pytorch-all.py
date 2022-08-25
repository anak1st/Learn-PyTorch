import time
import numpy as np
import torch


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


def cross_product(size, device):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    a = torch.tensor(a, device=device)
    b = torch.tensor(b, device=device)
    start = time.perf_counter()
    c = a @ b
    end = time.perf_counter()
    cost = end - start
    print(f"[{device}] time: {cost:.5f}(s)")
    return cost


def loop_test(times, size, device):
    t = 0
    for i in range(0, times):
        t += cross_product(size, device)
    t /= times
    print(f"[{device}] average time: {t:.5f} s")


if __name__ == '__main__':
    loop_test(100, 2000, gpu)
    loop_test(100, 2000, cpu)
