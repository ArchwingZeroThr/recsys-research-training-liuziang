"""
案例：
    演示张量的拼接操作。
涉及到的API：
    cat()
    不改变维度数，拼接张量，除了拼接的那个维度外，其它维度数必须保持一致。
    stack()
    会改变维度数，拼接张量，所有的维度都必须保持一致，
"""
# 导包
import torch

# 1. 创建两个张量。
t1 = torch.randint(1, 10, (2, 3))
print(f't1: {t1}, shape: {t1.shape}')

t2 = torch.randint(1, 10, (2, 3))
print(f't2: {t2}, shape: {t2.shape}')

# 2. 演示张量的拼接。
# 思路1: cat() 拼接张量。
t3 = torch.cat([t1, t2], dim=0)
print(f't3: {t3}, shape: {t3.shape}')

t_temp=torch.randint(1, 10, (2, 8))

t4 = torch.cat([t1, t_temp], dim=1)
print(f't4: {t4}, shape: {t4.shape}')


#而stack()只能全是shape相同的tensor合并,相当于在打[]，会新增维度
t5 = torch.stack([t1, t2], dim=2)
print(f't5: {t5}, shape: {t5.shape}')
#(2,3)+(2,3)=(2,2,3)