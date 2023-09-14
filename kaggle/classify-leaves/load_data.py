import collections
import math
import os
import shutil
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn

data_dir = "D:/Downloads/Data/Kaggle/classify-leaves"


def read_csv_labels(fname):
    """读取 `fname` 来给标签字典返回一个文件名。"""
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


labels = read_csv_labels(os.path.join(data_dir, 'train.csv')) # 存放训练集标签的文件


def copyfile(filename, target_dir):
    """将文件复制到目标目录。"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


def reorg_train_valid(data_dir, labels, valid_ratio):
    # 下面的collections.Counter就是统计label这个字典中有几个类别（返回字典）；.most_common()则转换成元组；[-1][1]则返回最后一个元组的第二个值(因为这个类别数量最小)
    n = collections.Counter(labels.values()).most_common()[-1][1] # n就是数量最少类别对应的数量
    n_valid_per_label = max(1, math.floor(n * valid_ratio)) # 根据最小类别数量，得出验证集的数量
    label_count = {}
    for train_file in labels: # 返回训练集中的图片名字列表(我们看到，训练标签转换成的字典，key就是训练集的图片名字)
        label = labels[train_file] # 每张图片 对应的标签
        fname = os.path.join(data_dir, train_file) # 每个图片的完整路径
        # 将图片复制到指定的目录下，这个是为了交叉验证使用，这里和训练集没区别
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label)) 
        # 制作验证集。注：标签名作为key,value为每个标签的图片数量
        if label not in label_count or label_count[label] < n_valid_per_label: 
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1 # 统计每个标签的图片数量
        else: # 制作训练集
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))
    return n_valid_per_label # 返回验证集的数量


# 在预测期间整理测试集，以方便读取
def reorg_test(data_dir):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    for test_file in test['image']: # 获取测试集图片的名字，复制到指定文件夹下
        copyfile(os.path.join(data_dir, test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))


# 调用前面定义的函数，进行整理数据集
def reorg_leave_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'train.csv')) # 是个字典
    reorg_train_valid(data_dir, labels, valid_ratio) # 生成训练集和验证集
    reorg_test(data_dir) # 生成测试集


batch_size = 128
valid_ratio = 0.1 # 验证集的比例
if not os.path.exists(data_dir + "\\" + "train_valid_test"): # 判断是否已经制作好了数据集
    print("start prepare img!")
    reorg_leave_data(data_dir, valid_ratio)
else:
    print("Already exists img!")
print('finish prepare img!')


transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0), # 从图片的中心点剪裁出24*24的图片
                                             ratio=(3.0 / 4.0, 4.0 / 3.0)),
    torchvision.transforms.RandomHorizontalFlip(), # 左右翻转
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4),
    # 加入随机噪音
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], # 对图片的每个通道做均值和方差
                                     [0.229, 0.224, 0.225])])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    # 加入随机噪音
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], # 依然 对图片的每个通道做均值和方差
                                     [0.229, 0.224, 0.225])])


# ImageFolder按照文件夹顺序，进行打标签

# 训练集和交叉验证集
train_ds, train_valid_ds = [ 
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train) for folder in ['train', 'train_valid']]

# 验证集和测试集
valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test) for folder in ['valid', 'test']]

# 把前面的数据放入torch的DataLoader，则每次迭代时，读取一个batch
train_iter, train_valid_iter = [
    DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)]

valid_iter = DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
