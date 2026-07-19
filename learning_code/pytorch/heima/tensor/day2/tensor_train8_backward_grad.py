"""
回顾：
权重更新公式：
    W新=W旧-学习率*梯度（学习率自己设置）
    梯度=损失函数的导数
    关于损失函数的导数，无需我们手动计算，因为非常常用，所以PyTorch模块内置有自动微分模块，
    专门实现针对于不同的损失函数求导，从而实现结合反向传播，更新权重参数w科偏置参数b

细节：
    只有标量张量才能求导，大多是float
"""

import torch

#1．定义变量，记录：初始的权重w（旧）
#参1：初始位，参2:是否自动微分〔求导〕，参3：数据类型
w= torch.tensor(10,requires_grad=True,dtype=torch.float)

#2.定义loss变量，表示损失函数
loss=2*w**2

#3.打印梯度函数类型
print(f'梯度函数类型:{type(loss.grad_fn)}')

#4.计算梯度，梯度=损失函数的导数，计算完毕后会记录到grad的属性中
# loss.backward()         #这里由于loss本身是标量可以不加sum()
loss.sum().backward()   #保证loss是1个标量，标准写法

#5.带入权重更新公式
w.data-=0.01*w.grad

#6.打印最终结果
print(f'w: {w.data}')