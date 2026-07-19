"""
案例：
    演示张量的基本运算.
涉及到的API：
    add()，sub()，mul()，div()，neg()    →加减乘除，取反，substract，multiply，divide
    add_()，sub_()，mul_()，div_()，neg_()→功能同上，只不过可以修改源数据，类似于Pandas部分的inplace=True
需要你记忆的：
    + - * /
"""
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

c = a.add(b)          # tensor([5., 7., 9.])，效果同+
d = a.sub(b)          # tensor([-3., -3., -3.])
e = a.mul(b)          # tensor([ 4., 10., 18.])
f = a.div(b)          # tensor([0.25, 0.4, 0.5])
g = a.neg()           # tensor([-1., -2., -3.])

print(a)              # 原张量 a 不变：tensor([1., 2., 3.])

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

x.add_(y)            # x 变为 tensor([5., 7., 9.])，效果同+=
print(x)             # 原 x 已被修改