import time
import numpy as np
import torch
import torch.nn as nn


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pytorch_gpu(n):
    a = np.random.rand(n, n)
    b = np.random.rand(n, n)
    a = torch.tensor(a, device=gpu)
    b = torch.tensor(b, device=gpu)
    start = time.perf_counter()
    c = a @ b
    end = time.perf_counter()
    print("Elapsed time is {:.5f} s".format(end - start))
    if n < 10:
        print(c)


if __name__ == '__main__':
    pytorch_gpu(15000)
