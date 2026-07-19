"""
案例：
    演示自动微分模块，循环实现计算梯度，更新参数。
需求：
    求y=X**2+20的极小值点并打印y是最小值时w的值（梯度）
解题步骤：
    1.定义点x=10 requires_grad=Truedtype=torch.float32
    2.定义函数y=X**2 +20
    3.利用梯度下降法循环迭代1000求最优解
    3.1正向计算（前向传播）
    3.2梯度清零x.grad.zero_()
    3.3反向传播
    3.4梯度更新x.data=X.data-0.01* x.grad
"""

import torch

w=torch.tensor(10,requires_grad=True,dtype=torch.float32)

loss=w**2+20
print(f'开始权重初位：{w}，Co.01 *w.grad)：无， loss:{loss}')

#利用梯度下降法，循环迭代求最优解
for i in range(1,101):
    # 正向计算（前向传播）
    loss=w**2+20

    # 3.2梯度清零x.grad.zero_()  默认梯度会累加
    #至此（第一次的时候），还没有计算梯度，所以w.grad=NONE,要做非空判断
    if w.grad is not None:
        w.grad.zero_()

    # 3.3反向传播（此处开始算梯度）
    loss.sum().backward()

    # 3.4梯度更新x.data=X.data-0.01* x.grad
    # print(f"梯度值为：{w.grad}")
    w.data=w.data-0.01*w.grad
    # print(f'第{i}次,w:{w},w.grad:{w.grad},loss:{loss}')

print(f'最终w:{w:.5f},w.grad:{w.grad:.5f},loss:{loss:.5f}')