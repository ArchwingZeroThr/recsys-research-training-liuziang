# Pytorch

# 一、前置条件

1. anaconda的安装

2. 通过anaconda安装构建出不同的环境，为后续不同的代码复现场景做准备。

![image\.png](../assets/pytorch/image.png)

- 此处若太慢，可以在base环境配置mamba，提升pytorch和后续更新速度

- 连接见:https://github\.com/mamba\-org/micromamba\-releases/releases/latest

- 下载对应版本的micromamba\.exe文件即可

- 最后将其配置到环境变量中

3. 在pytorch环境下配置jupyter noterbook

# 二、两大法宝函数

- `dir()`函数，能让我们知道工具箱以及工具箱中的分隔区有什么东西。

```Python
dir(pytorch)
```

- `help()`函数，能让我们知道每个工具是如何使用的，工具的使用方法。

```Python
help(pytorch.3)
```

# 三、Pytorch加载数据初认识

## 1\.Dataset\(从垃圾中发现可回收垃圾\)

##### （1）定义

提供一种方式去获取数据及其label:

- 获取每一个数据及其label

- 告诉我们总共有多少的数据

##### （2）实验过程——Dataset类代码实战！！！

1. 首先在Pycharm中配置python解释器（pytorch环境）

![image\.png](../assets/pytorch/image%201.png)

2. 随后在项目中输入代码

```Python
from torch.utils.data import Dataset
#此时使用jupyter中使用help(Dataset)查看作用,也可以ctrl+左键
```

![image\.png](../assets/pytorch/image%202.png)

3. 下载数据集蚂蚁蜜蜂
	详细代码见`learning_code/pytorch`






## 2\.Dataloader\(打包\)

为网络提供不同的数据形式 

