import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from net_module import Net
from torch.utils.tensorboard import SummaryWriter

# 超参数
learning_rate = 0.01
batch_size = 64
epochs = 15
path = './save'

train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Net()
# print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

def train():
    writer = SummaryWriter(path)
    writer.add_graph(net, input_to_model=torch.rand(64, 1, 28, 28))

    net.to(device)
    print('training on', device)
    for epoch in range(epochs):
        correct = 0.0
        total_loss = 0.0
        total = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)#第0维是batch_size，第1维是通道数，第2维是高，第3维是宽
            labels = labels.to(device)#第0维是batch_size
            # print(i)
            # print(images.shape)
            # print(labels.shape)

            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            #每个i循环里的loss加起来，最后除以循环次数也就是train_loader，得到平均loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            #max函数用来返回每一行的最大值，第0个返回值是最大值，第1个返回值是最大值的索引
            # print(predicted)
            # predicted是一个tensor，里面有64个预测值，每个值是0-9的数字，代表预测的数字

            correct += torch.eq(predicted, labels).sum().item()
            # print(torch.eq(predicted, labels).sum().item())
            #eq函数是用来比较预测值索引和标签是否符合，返回一个bool类型的tensor，sum函数是用来统计True的个数，item函数是用来返回tensor里的元素值
            total += labels.shape[0]
            #labels.shape[0]是batch_size,每次i循环处理64个样本，所以total每次加64，也可以理解成每次加batch_size个样本，最后total是60000
            #但是不能写成total += batch_size因为最后一次的batch_size不一定是64，可能是60000%64=16

        avg_loss = total_loss / len(train_loader)
        acc = correct / total
        writer.add_scalar('training loss', avg_loss, epoch+1)
        writer.add_scalar('training acc', acc, epoch+1)
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy:{:.4f}'.format(epoch + 1, epochs, avg_loss, acc))
    writer.close()

def test():
    net.to(device)
    print('testing on', device)
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = outputs.max(1)
            correct += torch.eq(predicted, labels).sum().item()
            total += labels.shape[0]
        acc = correct / total
        print('Test Accuracy:{:.4f}'.format(acc))

train()
test()

