# 一、前置条件

1. anaconda的安装
    
2. 通过anaconda安装构建出不同的环境，为后续不同的代码复现场景做准备。
    

![](https://cwafj6ayym1.feishu.cn/space/api/box/stream/download/asynccode/?code=YmVlNGU0ZWYyNmViMzc3MDdlN2IyYmUxMjkzNjhkMDNfa0ZpYmY5SlBybG5tVVliMHF5aDZqMksxNGdORFNHSDZfVG9rZW46S3JnYmJ3WXZHbzBCTFN4dE9XNmM4ZDFTblhlXzE3ODQ0NjE2Nzk6MTc4NDQ2NTI3OV9WNA&add_watermark=true&scene_type=CCM)

- 此处若太慢，可以在base环境配置mamba，提升pytorch和后续更新速度
    
- 连接见:https://github.com/mamba-org/micromamba-releases/releases/latest
    
- 下载对应版本的micromamba.exe文件即可
    
- 最后将其配置到环境变量中
    

3. 在pytorch环境下配置jupyter noterbook
    

# 二、两大法宝函数

- `dir()`函数，能让我们知道工具箱以及工具箱中的分隔区有什么东西。
    

```Python
dir(pytorch)
```

- `help()`函数，能让我们知道每个工具是如何使用的，工具的使用方法。
    

```Python
help(pytorch.3)
```

---

# 小土堆教程（不适合，暂放）

## 1.Dataset(从垃圾中发现可回收垃圾)

##### （1）定义

提供一种方式去获取数据及其label:

- 获取每一个数据及其label
    
- 告诉我们总共有多少的数据
    

##### （2）实验过程——Dataset类代码实战！！！

1. 首先在Pycharm中配置python解释器（pytorch环境）
    

![](https://cwafj6ayym1.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjJmNDE4YjJkODY3MGI2YmQ2OTczZTc0YzdhY2YyNjdfMzN6dlB4OGNnUjUzMnd0SkxROU9ZT1JhRERpdEVoN1FfVG9rZW46VHhYVmJya1BKbzYxVk94RHU3d2NwSnloblBiXzE3ODQ0NjE2Nzk6MTc4NDQ2NTI3OV9WNA&add_watermark=true&scene_type=CCM)

2. 随后在项目中输入代码
    

```Python
from torch.utils.data import Dataset
#此时使用jupyter中使用help(Dataset)查看作用,也可以ctrl+左键
```

![](https://cwafj6ayym1.feishu.cn/space/api/box/stream/download/asynccode/?code=Zjc5ZGExZjk5YjgzNTA1MGVkNWIyNmZiM2E1NjlkZGRfR1FtajRjbVlzRjNSZTZ4b1NKM1ViZ003MjFxRlN6SFVfVG9rZW46UTM4RWJseDFMb2laTzV4RHdiemN2ZVJCbkZoXzE3ODQ0NjE2Nzk6MTc4NDQ2NTI3OV9WNA&add_watermark=true&scene_type=CCM)

3. 下载数据集蚂蚁蜜蜂
    

  

  

  

## 2.Dataloader(打包)

为网络提供不同的数据形式

---

# 黑马教程

> 本笔记基于 PyTorch 框架，系统梳理张量的创建、类型、运算、索引、变形、拼接、自动微分及优化循环。所有代码均附带预期输出，适合速查与复习。

## 一、张量概述

  

- **定义**：张量是 PyTorch 中存储数据的容器，类似 NumPy 的 ndarray，但支持 GPU 加速和自动求导。
    
- **特点**：存储同一类型数值元素（整数、浮点数等），可为标量、向量、矩阵及更高维。
    
- **重要性**：所有深度学习模型（ANN、CNN、RNN 等）底层数据处理都基于张量。
    

## 二、张量的创建方式

### 1. `torch.tensor()` —— 根据已有数据创建

```Python
import torch

# 标量
t1 = torch.tensor(10)
print(t1)
# 输出: tensor(10)

# 二维张量（列表）
data = [[1, 2, 3], [4, 5, 6]]
t2 = torch.tensor(data)
print(t2)
# 输出: tensor([[1, 2, 3],
#               [4, 5, 6]])

# 从 NumPy 创建并指定类型
import numpy as np
data_np = np.random.randint(1, 10, size=(2, 3))
t3 = torch.tensor(data_np, dtype=torch.float)
print(t3)
# 输出示例（随机）: tensor([[3., 9., 6.],
#                          [4., 7., 1.]])
```

  

> **注意**：`torch.tensor` 只能根据**具体数据**创建，不能直接传入形状（如 `torch.tensor(2,3)` 会报错）。

  

### 2. `torch.Tensor()` —— 根据形状创建（也可传数据）

```Python
# 创建形状为 (10,) 的未初始化向量
t1 = torch.Tensor(10)
print(t1)
# 输出: tensor([0., 0., ..., 0.])   # 内容可能为极小随机值，通常显示为0

# 创建形状 (2,3)
t2 = torch.Tensor(2, 3)
print(t2)
# 输出: tensor([[0., 0., 0.],
#               [0., 0., 0.]])

# 也可传入数据
t3 = torch.Tensor([[1, 2], [3, 4]])
print(t3)
# 输出: tensor([[1., 2.],
#               [3., 4.]])
```

> **区别**：
> 
> - 小写 `torch.tensor`：只能从现有数据创建。
>     
> - 大写 `torch.Tensor`：既可根据形状创建（未初始化），也可根据数据创建。
>     

  

### 3. 指定类型的创建函数（不常用）

```Python
torch.IntTensor([1,2,3])      # tensor([1,2,3], dtype=torch.int32)
torch.FloatTensor([[1,2],[3,4]]) # tensor([[1.,2.],[3.,4.]])
```

  

### 4. 创建全0、全1、指定值张量

```Python
# 全1
ones = torch.ones(2, 3)
print(ones)
# 输出: tensor([[1., 1., 1.],
#               [1., 1., 1.]])

# 仿照形状
ones_like = torch.ones_like(ones)

# 全0
zeros = torch.zeros(2, 3)
# 全指定值
full = torch.full(size=(2,3), fill_value=255)
print(full)
# 输出: tensor([[255., 255., 255.],
#               [255., 255., 255.]])
```

  

### 5. 创建线性序列张量

```Python
# arange: start, end (exclusive), step
t1 = torch.arange(1, 10, 2)
print(t1)   # tensor([1, 3, 5, 7, 9])

# linspace: start, end (inclusive), number of points
t2 = torch.linspace(1, 10, 5)
print(t2)   # tensor([ 1.0000,  3.2500,  5.5000,  7.7500, 10.0000])
```

  

### 6. 创建随机张量

```Python
torch.manual_seed(42)   # 固定随机种子

# 均匀分布 [0,1)
rand_uniform = torch.rand(2, 3)
print(rand_uniform)
# 输出: tensor([[0.8823, 0.9150, 0.3829],
#               [0.9593, 0.3904, 0.6009]])

# 标准正态分布
rand_normal = torch.randn(2, 4)
# 随机整数 [low, high)
rand_int = torch.randint(0, 10, size=(2, 3))
print(rand_int)
# 输出: tensor([[6, 8, 3],
#               [2, 6, 4]])
```

  

### 7. 类型转换

  

```Python
t = torch.tensor([1,2,3], dtype=torch.float)
t_int16 = t.type(torch.int16)
print(t_int16)   # tensor([1, 2, 3], dtype=torch.int16)

# 其他: t.half(), t.double(), t.long(), t.short() 等
```

  

## 三、张量与 NumPy 互转

### 1. 张量 → NumPy（共享内存）

```Python
t = torch.tensor([1,2,3,4,5,6])
n = t.numpy()          # 共享内存
n[0] = 100
print(t)               # tensor([100, 2, 3, 4, 5, 6])   ← t 也被修改

# 避免共享：使用 .copy()
n2 = t.numpy().copy()
```

### 2. NumPy → 张量

```Python
n = np.array([1,2,3,4,5,6])
t1 = torch.tensor(n)        # 不共享内存
t2 = torch.from_numpy(n)    # 共享内存（推荐，效率高）
```

### 3. 标量张量提取 Python 值

```Python
scalar = torch.tensor(3.14159)
value = scalar.item()
print(value)   # 3.14159
```

  

## 四、张量的基本运算（元素级）

```Python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)   # tensor([5., 7., 9.])
print(a - b)   # tensor([-3., -3., -3.])
print(a * b)   # tensor([ 4., 10., 18.])
print(a / b)   # tensor([0.2500, 0.4000, 0.5000])
print(-a)      # tensor([-1., -2., -3.])

# 原地操作（带 _）
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
x.add_(y)      # x 变为 tensor([5., 7., 9.])
print(x)
```

  

## 五、点乘与矩阵乘法

### 1. 点乘（对应元素相乘）—— 形状相同

```Python
t1 = torch.tensor([[1,2,3],[4,5,6]])
t2 = torch.tensor([[7,8,9],[10,11,12]])
result = t1 * t2   # 或 t1.mul(t2)
print(result)
# 输出: tensor([[ 7, 16, 27],
#               [40, 55, 72]])
```

### 2. 矩阵乘法（点积）—— 列数 = 行数

```Python
A = torch.tensor([[1,2,3],[4,5,6]])     # (2,3)
B = torch.tensor([[7,8],[9,10],[11,12]]) # (3,2)
C = A @ B   # 或 A.matmul(B)
print(C)
# 输出: tensor([[ 58,  64],
#               [139, 154]])
```

  

## 六、常用数学函数

### 聚合函数（支持 dim 参数）

```Python
t = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float)
print(t.sum(dim=0))   # 列和 -> tensor([5., 7., 9.])
print(t.sum(dim=1))   # 行和 -> tensor([ 6., 15.])
print(t.sum())        # 全局和 -> tensor(21.)

print(t.mean(dim=0))  # 列均值 -> tensor([2.5, 3.5, 4.5])
```

### 其他函数（无 dim）

```Python
t.pow(2)      # 平方（或 t**2）
t.sqrt()      # 开方
t.exp()       # 指数
t.log()       # 自然对数
```

  

## 七、张量索引（Indexing）

用固定张量演示：

```Python
t1 = torch.tensor([
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
])

# 简单索引
print(t1[1])          # 第2行 -> tensor([ 6,  7,  8,  9, 10])
print(t1[:, 2])       # 第3列 -> tensor([ 3,  8, 13, 18, 23])

# 列表索引（花式）
print(t1[[1,3], [2,4]])   # (1,2)和(3,4) -> tensor([ 8, 20])

# 范围索引（切片）
print(t1[:3, :2])     # 前3行前2列
# 输出: tensor([[ 1,  2],
#               [ 6,  7],
#               [11, 12]])

# 步长
print(t1[::2, 1::2])  # 行步长2，列从1开始步长2
# 输出: tensor([[ 2,  4],
#               [12, 14],
#               [22, 24]])

# 布尔索引
print(t1[t1[:,2] > 5])   # 第三列 >5 的所有行

# 多维索引 (3D)
t2 = torch.randint(1,10,(3,4,5))
print(t2[0, :, :])    # 第0个二维切片
print(t2[:, 0, :])    # 所有第0行
print(t2[:, :, 0])    # 所有第0列
```

  

## 八、张量形状变换（Reshape）

```Python
t = torch.arange(1, 13)   # tensor([ 1,  2, ..., 12])

# reshape
t_reshaped = t.reshape(3, 4)
# 输出: tensor([[ 1,  2,  3,  4],
#               [ 5,  6,  7,  8],
#               [ 9, 10, 11, 12]])

# view（要求连续，否则用 reshape）
t_view = t.view(2, 6)

# 展平
t_flat = t_reshaped.flatten()   # 或 reshape(-1)

# 转置（二维）
t_matrix = torch.tensor([[1,2,3],[4,5,6]])
print(t_matrix.t())
# 输出: tensor([[1, 4],
#               [2, 5],
#               [3, 6]])

# 增加/删除维度
t = torch.randint(1,10,(4,3))
print(t.shape)   # torch.Size([4, 3])
t_unsqueeze = t.unsqueeze(1)   # (4,1,3)
t_squeeze = t_unsqueeze.squeeze(1)  # (4,3)

# 维度交换（permute）
t3d = torch.randint(1,10,(2,3,4))
t_permuted = t3d.permute(2,0,1)   # (4,2,3)
```

  

## 九、张量拼接（cat / stack）

### 1. `cat` —— 不增加维度，拼接维度需匹配

```Python
t1 = torch.randint(1,10,(2,3))
t2 = torch.randint(1,10,(2,3))
print(t1.shape)   # (2,3)
print(t2.shape)   # (2,3)

# 沿 dim=0（行）拼接，结果 (4,3)
t3 = torch.cat([t1, t2], dim=0)
print(t3.shape)   # torch.Size([4, 3])

# 沿 dim=1（列）拼接，要求除拼接维外其他一致
t_temp = torch.randint(1,10,(2,8))   # 列数为8
t4 = torch.cat([t1, t_temp], dim=1)  # 结果 (2, 3+8=11)
print(t4.shape)   # torch.Size([2, 11])
```

  

### 2. `stack` —— 增加新维度，所有形状必须完全相同

```Python
t1 = torch.randint(1,10,(2,3))
t2 = torch.randint(1,10,(2,3))
# 沿 dim=0 堆叠，结果 (2,2,3)
t5 = torch.stack([t1, t2], dim=0)
print(t5.shape)   # torch.Size([2, 2, 3])

# 沿 dim=2 堆叠，结果 (2,3,2)
t6 = torch.stack([t1, t2], dim=2)
print(t6.shape)   # torch.Size([2, 3, 2])
```

> **区别**：`cat` 不增加总维度数，`stack` 会增加一个维度（相当于把张量列表当作新维度）。

  

## 十、自动微分与梯度更新（核心）

### 1. 基础梯度计算

```Python
# 定义需要求导的张量（requires_grad=True）
w = torch.tensor(10.0, requires_grad=True)

# 定义损失函数（标量）
loss = 2 * w ** 2

# 反向传播计算梯度
loss.backward()   # 或 loss.sum().backward() 确保标量

# 查看梯度
print(w.grad)     # tensor(40.)  因为 d(2w^2)/dw = 4w = 40

# 更新权重（手动）
w.data -= 0.01 * w.grad
print(w.data)     # tensor(9.6000)
```

> **注意**：`loss.backward()` 要求 `loss` 是标量，若为非标量需先 `sum()`。

  

### 2. 梯度清零（重要）

梯度会累加，每次反向传播前需清零：

```Python
if w.grad is not None:
    w.grad.zero_()
```

  

## 十一、优化循环实战（求极小值）

  

**需求**：求函数 `y = x² + 20` 的极小值点，初始 `x=10`，学习率 0.01，迭代 100 次。

```Python
import torch

x = torch.tensor(10.0, requires_grad=True)

for i in range(1, 101):
    # 前向传播
    loss = x**2 + 20

    # 梯度清零
    if x.grad is not None:
        x.grad.zero_()

    # 反向传播
    loss.sum().backward()   # loss是标量，可省略 .sum()

    # 更新参数（梯度下降）
    x.data -= 0.01 * x.grad

    # 可选打印
    if i % 20 == 0:
        print(f"第{i:3d}次: x={x.item():.5f}, loss={loss.item():.5f}")

# 最终结果
print(f"最终 x={x.item():.5f}, 梯度={x.grad.item():.5f}, loss={loss.item():.5f}")
```

  

**预期输出**（因随机种子固定，结果稳定）：

```Plain
第 20次: x=0.40960, loss=20.16777
第 40次: x=0.01678, loss=20.00028
第 60次: x=0.00069, loss=20.00000
第 80次: x=0.00003, loss=20.00000
第100次: x=0.00000, loss=20.00000
最终 x=0.00000, 梯度=0.00000, loss=20.00000
```

  

> 可见 `x` 收敛到 0，达到极小值 20，验证了梯度下降的有效性。

  

## 十二、总结与速查表

|                  |                               |                     |
| ---------------- | ----------------------------- | ------------------- |
| 操作分类             | 推荐API                         | 说明                  |
| **创建（有数据）**      | `torch.tensor(data)`          | 最常用                 |
| **创建（指定形状）**     | `torch.Tensor(2,3)`           | 未初始化                |
| **全0/全1/指定值**    | `zeros`, `ones`, `full`       | 带 `_like` 仿照形状      |
| **线性序列**         | `arange`, `linspace`          | -                   |
| **随机**           | `randint`, `rand`, `randn`    | 用 `manual_seed` 固定  |
| **类型转换**         | `.type()` 或 `.to()`           | -                   |
| **转** **NumPy**  | `.numpy()`                    | 共享内存，用 `.copy()` 避免 |
| **NumPy** **回转** | `torch.tensor(n)`             | 不共享；`from_numpy` 共享 |
| **标量取值**         | `.item()`                     | 仅单个元素               |
| **元素运算**         | `+ - * /`                     | 原地 `_` 版本           |
| **矩阵乘法**         | `@` 或 `.matmul()`             | -                   |
| **聚合统计**         | `sum`, `mean`, `max`, `min`   | 指定 `dim`            |
| **索引**           | `[行,列]`、切片、布尔                 | -                   |
| **形状变化**         | `.reshape()`                  | 最通用                 |
| **展平**           | `.flatten()` 或 `.reshape(-1)` | -                   |
| **维度交换**         | `.permute()`                  | -                   |
| **拼接**           | `cat`（不增维）、`stack`（增维）        | 注意维度匹配              |
| **自动微分**         | `.backward()`                 | 对标量使用               |
| **梯度清零**         | `.grad.zero_()`               | 每次反向传播前             |
| **参数更新**         | `data -= lr * grad`           | 手动更新                |