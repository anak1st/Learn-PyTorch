import torch
import torch.backends.cudnn as cudnn


def test_cuda():
    ok = torch.cuda.is_available()
    print("PyTorch with CUDA is available:{}".format(ok))
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU:{}".format(gpu_name))
    cuda_v = torch.version.cuda
    print("CUDA:{}".format(cuda_v))
    cudnn_v = cudnn.version()
    print("cuDNN:{}".format(cudnn_v))


if __name__ == "__main__":
    test_cuda()
