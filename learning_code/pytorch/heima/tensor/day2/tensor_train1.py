"""
1.张量转numpy:
    使用Tensor.numpy函数可以将张量转换为ndarray数组，但是共享内存，可以使用copy函数避免共享。
2.numpy转张量:
    使用torch.tensor可以将ndarray数组转换为Tensor,默认不共享内存

涉及到的API:
    场景1：张量→numpy nd数组对象
        张量对象.numpy()            共享内存
        张量对象.numpy().copy()     不共享内存，链式编程写法。
    场景2：numpy nd数组→张量
        from_numpy()               共享内存
        torch.tensor(nd数组)        不共享内存
    场景3：从标量张量中提取其内容
        标量张量.item()

掌握：
    张量——>numpy:     张量对象.numpy()
    numpy——>张量:     torch.tensor(nd数组)
    标量张量.item()
"""

import torch
import numpy as np
#场景1：张量——>numpy
def dm01():
    t1=torch.tensor([1,2,3,4,5,6])
    print(f"t1: {t1}, type(t1): {type(t1)}")
    n1=t1.numpy()#共享内存
    print(f"n1: {n1}, type(n1): {type(n1)}")
    print('-'*30)

    t2=torch.tensor([7,8,9,10,11,12])
    n2=t2.numpy().copy()
    #展示共享内存
    n1[0]=100
    print(f"n1: {n1}")
    print(f"t1: {t1}")
    print('-'*30)
    #展示不共享内存
    n2[0]=100
    print(f"n2: {n2}")
    print(f"t2: {t2}")

#场景2：numpy——>张量
def dm02():
    n1=np.array([1,2,3,4,5,6])
    print(f"n1: {n1}, type(n1): {type(n1)}")
    t1=torch.tensor(n1)
    print(f"t1: {t1}, type(t1): {type(t1)}")


#场景3：从标量张量（只有一个值的张量）中提取其内容
def dm03():
    # t1=torch.tensor([1,2,3,4,5,6])    无法提取向量,只能提取1个值
    t1=torch.tensor(1000)
    a=t1.item()
    print(a)


if __name__ == "__main__":
    # dm01()
    # dm02()
    dm03()