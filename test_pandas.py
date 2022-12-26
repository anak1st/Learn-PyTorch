import os
import pandas as pd

root = os.path.dirname(__file__)    # test_pandas
# root = os.path.dirname(root)      # root

data_folder = os.path.join(root, 'data')

print("data file in:", os.path.join(data_folder, 'house_tiny.csv'))
os.makedirs(data_folder, exist_ok=True)
data_file = os.path.join(data_folder, 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')   # 列名
    f.write('NA,Pave,127500\n')         # 每⾏表⽰⼀个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')


data = pd.read_csv(data_file)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)