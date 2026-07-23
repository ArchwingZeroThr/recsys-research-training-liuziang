"""案例：
演示自动微分的反文应用场景.
结论：
1. 先前向转插(正向传插) 计算出 换测值(z)
2. 基于损失函数，结台 预器值(z)和 真实值(y)，来计算 梯度.
3. 结合权重更新公式 w新 = W旧 - 学习率 * 梯度， 来更新 权重."""

import torch

#1. 定义x，表示：特征（输入数据），假设：2行5列，全1矩阵
x = torch.ones(2,5)

#2. 定义y，表示：标签（真实值），假设:2行3列，全0矩阵
y = torch.zeros(2,3)

#3. 初始化权重和偏置
w = torch.randn(5,3,requires_grad=True)
b = torch.randn(3,requires_grad=True)#按一行有多少元素算
print(f'w的初始{w}')
print(f'b的初始{b}')

#4. 基于前向传播
z= x@w + b
print(z)

#5. 定义损失函数，求导，结合反向传播，更新权重（已经更新）
criterion = torch.nn.MSELoss()
loss = criterion(z, y)  #loss=损失
loss.sum().backward()

#6. 打印更新后的wb
print(f'w的梯度{w.grad}')
print(f'b的梯度{b.grad}')