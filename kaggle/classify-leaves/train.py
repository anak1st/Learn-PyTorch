import torch
from torch import nn
import torchvision
from load_data import *


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate(net, data_iter, loss, device):  # 这个是在评估模型的时候使用
    los, acc, num = 0, 0, 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        
        y_hat = net(X)
        l = loss(y_hat, y)
        
        los += float(l.sum())
        acc += accuracy(y_hat, y)
        num += y.numel()
    return los / num, acc / num


def train_epoch(net, train_iter, loss, optimizer):
    los, acc, num = 0, 0, 0
    for i, (X, y) in enumerate(train_iter):
        X, y = X.to(device), y.to(device)
        
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        l.mean().backward()
        optimizer.step()

        los += float(l.sum())
        acc += accuracy(y_hat, y)
        num += y.numel()
    return los / num, acc / num


def train(net, train_iter, valid_iter, num_epochs, loss, lr, wd, device,
          lr_period, lr_decay):
    """
    wd：权衰量,用于防止过拟合
    lr_period：每隔几个epoch降低学习率
    lr_decay：降低学习率的比例
    """
    net.to(device)

    # 随机梯度下降
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     10,
                                                                     T_mult=2)

    # optimizer = torch.optim.SGD(
    #     (param for param in net.parameters() if param.requires_grad), lr=lr,
    #     momentum=0.9, weight_decay=wd)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    print("start train")
    for epoch in range(num_epochs):
        print(f"epoch [{epoch+1}], lr: {optimizer.state_dict()['param_groups'][0]['lr']:.3e}")
        train_loss, train_accuracy = train_epoch(net, train_iter, loss, optimizer)
        valid_loss, valid_accuracy = evaluate(net, valid_iter, loss, device)
        print(f'train loss: {train_loss:.3f}, acc: {train_accuracy * 100:.3f}%')
        print(f'valid loss: {valid_loss:.3f}, acc: {valid_accuracy * 100:.3f}%')

        scheduler.step()  # 一个epoch完了，衰减学习率
    pass


# 调用上面的函数：训练和验证模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = torchvision.models.resnet18(num_classes=176)

# Xavier Uniform模型初始化，在每一层网络保证输入和输出的方差相同
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)  # 应用Xavier Uniform初始化

loss = nn.CrossEntropyLoss(reduction='none')  # reduction='none'表示返回n个样本的loss

num_epochs, lr, wd = 10, 1e-4, 1e-4
lr_period, lr_decay = 2, 0.9

net_path = os.path.join(data_dir, 'model.pth')
if os.path.exists(net_path):
    net.load_state_dict(torch.load(net_path))
    print(f"load model from {net_path}")

train(net, train_iter, valid_iter, num_epochs, loss, lr, wd, device, lr_period, lr_decay)
train(net, train_valid_iter, valid_iter, num_epochs, loss, lr, wd, device, lr_period, lr_decay)

torch.save(net.state_dict(), net_path)

preds = []
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))  # 存放 测试集图片地址 的文件
for X, _ in test_iter:
    y_hat = net(X.to(device))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())
sorted_ids = test['image']  # 对应的id
df = pd.DataFrame({'image': sorted_ids, 'label': preds})  # 转换成pandas的DF格式

# .apply()函数：遍历DataFrame的元素（一行数据或者一列数据），默认列遍历
# ImageFolder返回对象的.classes属性：用一个 list 保存类别名称
# 这个的作用是：模型预测出来是概率最大的那个数的下标，在保存文件时，需要把数字类别转换为字符串类别，
# train_valid_ds.classes就是获取字符串类别名（返回的是一个列表），然后使用apply一行一行读取出来，把数字类别转换为字符串类别
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv(os.path.join(data_dir, 'submission.csv'), index=False)
