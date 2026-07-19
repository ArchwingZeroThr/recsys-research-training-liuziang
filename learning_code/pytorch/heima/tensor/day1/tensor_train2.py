"""
案例：
    演示如何创建全0，全1，指定值的张量·
涉及到的函数如下：
    torch.ones和torch.ones_like创建全1张量
    torch.zeros和torch.zeros_like创建全o张量
    torch.full和torch.full_like创建全为指定值张量
需要掌握:
    zeros(),full()
"""
#alt+回车导包
import torch


#场景1：torch.ones和torch.ones_like创建全1张量
t1=torch.ones(2,3)
print(f't1: {t1}, type(t1): {type(t1)}')
print('-'*20)

t2=torch.ones(3,4)
print('-'*20)

#t3基于t2的形状创建全1张量,ones_like的输入需要tensor
t3=torch.ones_like(t2)
print(f't3: {t3}, type(t3): {type(t3)}')
print('-'*20)

#场景2：torch.zeros和torch.zeros_like创建全o张量
# 同场景1


#场景3：torch.full和torch.full_like创建全为指定值张量
t4=torch.full(size=(2,3),fill_value=255)
print(f't4: {t4}, type(t4): {type(t4)}')
print('-'*20)

t5=torch.ones(3,4)
t6=torch.full_like(t5,255)
print(f't6: {t6}, type(t6): {type(t6)}')