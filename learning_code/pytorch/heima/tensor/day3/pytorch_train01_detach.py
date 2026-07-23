"""
案例：
    演示 detach()函数的功能，解决 自动微分的弊端.
回顾：
    自动微分 =求导，即：基于损失函数，计算梯度.
    结合权重更新公式：W新 = W旧 - 学习率 * 梯度，来更新权重的.
问题：
    一个张量一旦设置了自动微分，这个张量就不能直接转成 numpy的 ndarray对象了，需要通过 detach()函数解决
"""
import torch
import numpy as np
from sympy import true

t1 = torch.tensor([1,2,3,4,5,6],requires_grad=True,dtype=torch.float)
print(f't1:{type(t1)}')

#未使用detach()前转numpy，此时requires_grad不可设置，一旦设置不可转
# n1 = t1.numpy()
# print(f'n1:{type(n1)}')

#此时使用detach函数拷贝一份张量然后转换
t2 = t1.detach()
print(f't2:{type(t2)}')

#测试t1和t2是否共享同一空间
t1.data[0]=100
print(f't2:{t2}')
#t2:tensor([100.,   2.,   3.,   4.,   5.,   6.])说明共享

# 5. 查看t1 和 t2谁可以自动微分
print(f't1: {t1.requires_grad}')
print(f't2: {t2.requires_grad}')

#把t2转numpy
n1=t2.numpy()
print(f'n1:{n1}')