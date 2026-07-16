"""
案例:
    如何创建线性和随机张量
涉及到的函数：
    torch.arange()和torch.linspace()创建线性张量
    torch.random.initial_seed()和torch. random. manual_seed()随机种子设置
    torch.rand/randn()创建随机浮点类型张量
    torch.randint(low，high，size=())创建随机整数类型张量
掌握的函数：
    arange(),Linspace()，manval_seea()，randint()
"""
import torch

# 1. 定义函数，演示：创建线性张量
def dm01():
    #场景1：指定范围的线性张量
    # 参1：起始值，参2：结束值，参3：步长。
    t1=torch.arange(1,10,2)
    print(f't1: {t1}, type(t1): {type(t1)}')
    print('-'*20)

    #场景2：指定范围的线性张量——>等差数列
    # 参1：起始值，参2：结束值，参3：元素的个数
    t2=torch.linspace(1,10,2)
    print(f't2: {t2}, type(t2): {type(t2)}')
    print('-'*20)


# 2. 定义函数，演示：创建随机张量
def dm02():
    #场景1：指定范围的线性张量
    # 参1：起始值，参2：结束值，参3：步长。
    t1=torch.arange(1,10,2)
    print(f't1: {t1}, type(t1): {type(t1)}')
    print('-'*20)

    #场景2：指定范围的线性张量——>等差数列
    # 参1：起始值，参2：结束值，参3：元素的个数
    t2=torch.linspace(1,10,2)
    print(f't2: {t2}, type(t2): {type(t2)}')
    print('-'*20)

# 3. 测试函数：
if __name__ == '__main__':
    # dm01()
    dm02()
    # pass