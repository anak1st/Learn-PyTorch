import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
from torch.utils.tensorboard import SummaryWriter   
import matplotlib.pyplot as plt


data_path = "D:/Downloads/Data/PyTorch/data"
writer = SummaryWriter('./logs/softmax')


def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root=data_path,
                                        train=True,
                                        transform=trans,
                                        download=True)
    mnist_test = datasets.FashionMNIST(root=data_path,
                                       train=False,
                                       transform=trans,
                                       download=True)
    return (DataLoader(mnist_train, batch_size, shuffle=True),
            DataLoader(mnist_test, batch_size, shuffle=False))


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 28 * 28
num_outputs = 10

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    net.eval()
    
    acc = 0
    num = 0

    with torch.no_grad():
        for X, y in data_iter:
            acc += accuracy(net(X), y)
            num += y.numel()
    return acc / num


def train_epoch(net, train_iter, loss, updater):
    net.train()

    los = 0
    acc = 0
    num = 0

    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        updater.zero_grad()
        l.mean().backward()
        updater.step()

        los += float(l.sum())
        acc += accuracy(y_hat, y)
        num += y.numel()
    return los / num, acc / num


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('train acc', train_acc, epoch)
        writer.add_scalar('test acc', test_acc, epoch)

        print(f"[epoch:{epoch}] loss:{train_loss:.4f}, ", end="")
        print(f"train acc:{train_acc*100:.2f}%, test acc:{test_acc*100:.2f}%")
    pass


num_epochs = 10
train(net, train_iter, test_iter, loss, num_epochs, trainer)
