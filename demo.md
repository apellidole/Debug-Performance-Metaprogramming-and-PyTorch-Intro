

创建一个张量
题目：使用 PyTorch 创建一个 2x3 的随机张量，并计算其元素的平均值。
解题方法：
(a) 使用 PyTorch 的 torch.rand 函数创建一个随机张量。
(b) 使用 tensor.mean() 方法计算张量元素的平均值。
(c) 打印张量、其形状及其平均值。

```python
创建一个张量
题目：使用 PyTorch 创建一个 2x3 的随机张量，并计算其元素的平均值。
解题方法：
(a) 使用 PyTorch 的 torch.rand 函数创建一个随机张量。
(b) 使用 tensor.mean() 方法计算张量元素的平均值。
(c) 打印张量、其形状及其平均值。
```

张量的高级运算
题目：创建两个 3x3 的张量，计算它们的逐元素加法、矩阵乘法以及逐元素的对
数和平方根，并打印结果。
解题方法：
(a) 使用 torch.tensor 创建两个张量。
(b) 使用 + 和 torch.matmul 计算张量的逐元素加法和矩阵乘法。
(c) 使用 torch.log 和 torch.sqrt 计算张量的逐元素对数和平方根。

```python
import torch  # 导入 PyTorch 库

# (a) 使用 torch.tensor 创建两个 3x3 的张量
tensor_a = torch.tensor([[1.0, 2.0, 3.0],
                         [4.0, 5.0, 6.0],
                         [7.0, 8.0, 9.0]])

tensor_b = torch.tensor([[9.0, 8.0, 7.0],
                         [6.0, 5.0, 4.0],
                         [3.0, 2.0, 1.0]])

# (b) 使用 + 和 torch.matmul 计算张量的逐元素加法和矩阵乘法
elementwise_addition = tensor_a + tensor_b
matrix_multiplication = torch.matmul(tensor_a, tensor_b)

# (c) 使用 torch.log 和 torch.sqrt 计算张量的逐元素对数和平方根
log_tensor_a = torch.log(tensor_a)
sqrt_tensor_a = torch.sqrt(tensor_a)

# 打印结果
print("张量 A:\n", tensor_a)
print("张量 B:\n", tensor_b)
print("逐元素加法:\n", elementwise_addition)
print("矩阵乘法:\n", matrix_multiplication)
print("张量 A 的逐元素对数:\n", log_tensor_a)
print("张量 A 的逐元素平方根:\n", sqrt_tensor_a)

```

使用 GPU 加速计算和内存管理
题目：检查环境中是否有可用的 GPU，并将张量从 CPU 移动到 GPU 上进行计
算，最后将结果移动回 CPU。
解题方法：
(a) 使用 torch.cuda.is_available() 检查是否有 GPU。
(b) 如果有可用 GPU，将张量移动到 GPU 上，使用 GPU 进行计算。
(c) 将计算结果从 GPU 移动回 CPU 并打印。

```python
import torch  # 导入 PyTorch 库

# (a) 使用 torch.cuda.is_available() 检查是否有 GPU
if torch.cuda.is_available():
    print("GPU is available. Using GPU for computation.")
    device = torch.device('cuda')  # 指定 GPU 设备
else:
    print("GPU is not available. Using CPU for computation.")
    device = torch.device('cpu')  # 指定 CPU 设备

# 创建一个张量并打印其所在设备
tensor_cpu = torch.tensor([[1.0, 2.0, 3.0], 
                           [4.0, 5.0, 6.0], 
                           [7.0, 8.0, 9.0]])
print("张量初始所在设备：", tensor_cpu.device)

# (b) 如果有可用 GPU，将张量移动到 GPU 上，使用 GPU 进行计算
tensor_gpu = tensor_cpu.to(device)
print("张量移动到设备：", tensor_gpu.device)

# 对张量进行计算（例如逐元素平方计算）
result_gpu = tensor_gpu ** 2

# (c) 将计算结果从 GPU 移动回 CPU 并打印
result_cpu = result_gpu.to('cpu')
print("计算结果（在 CPU 上）：\n", result_cpu)

```

. 创建一个简单的线性回归模型
题目：使用 PyTorch 实现一个简单的线性回归模型，生成随机数据并训练模型。
解题方法：
(a) 使用 torch.nn.Linear 创建线性模型。
(b) 生成随机数据，定义损失函数和优化器，进行模型训练。

```python
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 的神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块

# (a) 使用 torch.nn.Linear 创建线性模型
# 定义一个简单的线性回归模型，输入特征数为 1，输出特征数为 1
model = nn.Linear(1, 1)

# (b) 生成随机数据
# 生成 100 个数据点，x 是随机输入，y 是真实的线性输出
x = torch.rand(100, 1) * 10  # 输入 x，范围在 [0, 10]
y = 2 * x + 3 + torch.randn(100, 1) * 2  # 线性关系 y = 2x + 3，加上随机噪声

# 定义损失函数为均方误差 (MSE)
criterion = nn.MSELoss()

# 定义优化器为随机梯度下降 (SGD)，学习率为 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000  # 训练 1000 次迭代
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    predictions = model(x)
    
    # 计算损失
    loss = criterion(predictions, y)
    
    # 反向传播：计算梯度
    optimizer.zero_grad()  # 清除上一步的梯度
    loss.backward()  # 反向传播计算梯度
    
    # 更新模型参数
    optimizer.step()
    
    # 每 100 次迭代打印一次损失值
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印训练后的模型参数
print(f'训练后的模型权重: {model.weight.item():.4f}, 偏置: {model.bias.item():.4f}')

```



使用 DataLoader 加载数据
题目：创建一个自定义数据集类，并使用 PyTorch 的 DataLoader 进行批次化数
据加载。
解题方法：
(a) 定义一个自定义数据集类 SimpleDataset。
(b) 使用 DataLoader 对数据进行批量加载和迭代。

```python
import torch
from torch.utils.data import Dataset, DataLoader

# (a) 定义一个自定义数据集类 SimpleDataset
class SimpleDataset(Dataset):
    def __init__(self, num_samples):
        # 初始化数据集，这里生成一些简单的数据
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, 2)  # 生成 num_samples 个 2 维样本数据
        self.labels = torch.randint(0, 2, (num_samples,))  # 随机生成 0 和 1 两种类别标签

    def __len__(self):
        # 返回数据集的大小
        return self.num_samples

    def __getitem__(self, idx):
        # 根据索引返回数据和对应的标签
        return self.data[idx], self.labels[idx]

# 创建一个包含 100 个样本的数据集
dataset = SimpleDataset(num_samples=100)

# (b) 使用 DataLoader 对数据进行批量加载和迭代
# 使用 DataLoader 将数据集加载为每批次 10 个样本
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 迭代 DataLoader 并打印每个批次的数据
for batch_idx, (data, labels) in enumerate(dataloader):
    print(f'批次 {batch_idx + 1}:')
    print(f'数据: {data}')
    print(f'标签: {labels}\n')

```

