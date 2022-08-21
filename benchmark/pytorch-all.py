import time
import numpy as np
import torch
import torch.nn as nn


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pytorch_cpu(size, flag=False):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    a = torch.tensor(a)
    b = torch.tensor(b)
    start = time.perf_counter()
    c = a @ b
    end = time.perf_counter()
    cost = end - start
    if flag:
        print(f"CPU Elapsed time is {cost:.5f} s")
    return cost


def pytorch_gpu(size, flag=False):
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    a = torch.tensor(a, device=gpu)
    b = torch.tensor(b, device=gpu)
    start = time.perf_counter()
    c = a @ b
    end = time.perf_counter()
    cost = end - start
    if flag:
        print(f"GPU Elapsed time is {cost:.5f} s")
    return cost


if __name__ == '__main__':
    size = 5000
    cost_cpu = 0
    cost_gpu = 0
    times = 5
    for i in range(0, times):
        cost_cpu += pytorch_cpu(size)
        cost_gpu += pytorch_gpu(size, True)
    cost_cpu /= times
    cost_gpu /= times
    print(f"CPU Elapsed Average time is {cost_cpu:.5f} s")
    print(f"GPU Elapsed Average time is {cost_gpu:.5f} s")
