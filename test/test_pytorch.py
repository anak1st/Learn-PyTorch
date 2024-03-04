import torch
import torch.backends.cudnn as cudnn


def test_cuda():
    ok = torch.cuda.is_available()
    print(f"PyTorch with CUDA is available: {ok}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    cuda_v = torch.version.cuda
    print(f"CUDA: {cuda_v}")
    cudnn_v = cudnn.version()
    print(f"cuDNN: {cudnn_v}")


if __name__ == "__main__":
    test_cuda()
