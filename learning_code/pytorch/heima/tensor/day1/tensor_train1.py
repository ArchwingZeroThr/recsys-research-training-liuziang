"""
张量：
    PyTorCh框架属于最常用的深度学习框架，无论是后续要学的ANN（人工神经网络），还是CNN（卷积神经网络），RNN（循环神经网络）
    底层在处理数据时，都是使用张量来处理的。
    张量→存储同一类型元素的容器，且元素值必须是数值才可以。
张量的基本创建方式：
    torch.tensor根据指定数据创建张量——>tensor(值，类型dtype=float)
    torch.Tensor根据形状创建张量，其也可用来创建指定数据的张量
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor创建指定类型的张量

重点注意：
    Tensor根据形状创建张量，其也可用来创建指定数据的张量
    但小写用得多，3.几乎不用

"""
import torch
import numpy as np
from numpy.ma.core import size


# 1.定义函数，演示：torch.tensor根据指定数据创建张量
def dm01():
    #场景1: 标量张量
    t1 = torch.tensor(10)
    print(f"type = {t1.type()},t1 = {t1}")
    print('-'*30)

    #场景2: 二维张量
    data1 = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data1)
    print(f"type = {t2.type()},t2 = {t2}")
    print('-'*30)

    #场景3: 三维张量
    data2=np.random.randint(1,10,size=(2,3))
    t3 = torch.tensor(data2,dtype=torch.float)
    print(f"type = {data2.dtype},type = {t3.type()},t3 = {t3}")
    print('-'*30)

    #场景4: 尝试直接创建制定维度（例如2行3列）张量
    """
    会报错，小写tensor根据给出的数据创建，大写Tensor根据形状创建
    """
    # data3 = torch.tensor(2,3)
    # print(f"type = {data3.type()},data3 = {data3}")
"""
    大写Tensor和小写tensor的区别是什么
"""



# 2.定义函数，演示：torch.Tensor根据形状创建张量，其也可用来创建指定数据的张量
def dm02():
    #场景1: 标量张量
    t1 = torch.Tensor(10)
    print(f"type = {t1.type()},t1 = {t1}")
    print('-'*30)

    #场景2: 二维张量
    data1 = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data1)
    print(f"type = {t2.type()},t2 = {t2}")
    print('-'*30)

    #场景3: 三维张量
    data2=np.random.randint(1,10,size=(2,3))
    t3 = torch.Tensor(data2)
    print(f"type = {data2.dtype},type = {t3.type()},t3 = {t3}")
    print('-'*30)

    #场景4: 尝试直接创建制定维度（例如2行3列）张量
    """
    会报错，小写tensor根据给出的数据创建，大写Tensor根据形状创建
    """
    data3 = torch.Tensor(2,3)
    print(f"type = {data3.type()},data3 = {data3}")


# 3.定义函数，演示：torch.IntTensor、torch.FloatTensor、torch.DoubleTensor创建指定类型的张量

# 4.定义测试函数
if __name__ == '__main__':
    # dm01()
    dm02()