"""
案例：
    演示张量常用的运算函数.
涉及到的API(函数）如下：
    sum()，max()，min()，mean()  →都有dim参数，0表示列，1表示行
    pow()=**，sqrt()，exp()，log()，log2()，log10()    →没有dim参数
掌握的函数：
    sum()，max()，min()，mean()
"""

import torch

t1=torch.tensor([[1,2,3],[4,5,6]])
print(t1)
print('-'*30)

#有dim的函数
print(t1.sum(dim=0))    #列
print(t1.sum(dim=1))    #行
print(t1.sum())         #全局

#均值等需要float类型
t2=t1.type(torch.float)
print(t2.mean(dim=0))    #列
print(t2.mean(dim=1))    #行
print(t2.mean())         #全局

#无dim的函数，看看就好
