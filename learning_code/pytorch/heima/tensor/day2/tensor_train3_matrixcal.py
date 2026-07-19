"""
案例:
    演示张量的点乘和矩阵乘法操作。

点乘：
    要求：两个张量的维度保持一致，对应元素直接做相应的操作。
API:
    t1 * t2
    t1.mul(t2)  multiply：乘法


矩阵乘法：
    要求：两个张量，第一个张量的列数，等于第二个张量的行数(A列=B行）
结果：A行B列
API:
    t1 @ t2
    t1.matmul(t2)
"""
import torch

# 1.点乘
def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    t3 = t1 * t2
    print(t3)

# 2.矩阵乘法
def dm02():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[7,8],[9,10],[11,12]])
    t3 = t1 @ t2
    print(t3)

if __name__ == '__main__':
    dm01()
    dm02()