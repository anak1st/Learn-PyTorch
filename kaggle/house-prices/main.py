import os
import numpy as np
import pandas as pd
import torch
from torch import nn

data_root = "D:\Downloads\Data\Kaggle\house-prices-advanced-regression-techniques"

train_data = pd.read_csv(os.path.join(data_root, 'train.csv'))
test_data = pd.read_csv(os.path.join(data_root, 'test.csv'))

print(train_data.shape)
print(test_data.shape)