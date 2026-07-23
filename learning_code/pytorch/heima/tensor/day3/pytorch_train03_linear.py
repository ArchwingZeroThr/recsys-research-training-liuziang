"""
numpy——>tensor——>tensordataset——>dataloader
"""
import torch
from torch.utils.data import TensorDataset# 构造数据集对象
from torch.utils.data import DataLoader# 数据加载器
from torch import nn #nn模块中有平方损失函数和假设函数
from torch import optim # optim模块中有优化器函数
from sklearn.datasets import make_regression#创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']#用于正常显示中文标签
plt.rcParams['axes.unicode_minus']=False   #用于正常显示负号

# 1. 定义函数，创建线性回归样本数据
def create_dataset():
    # 1.创建数据集对象,x为特征,y为目标值
    x,y,coef = make_regression(
        n_samples=100,      #100样本点
        n_features=1,       #1个样本点
        coef=True,          #是否返问系数，默认为False，返回值为None
        noise=10,           #噪声，噪声越大，样本点超做，噪严越小，样本点超集中
        random_state=3,      #随机种子，随机种子相同，输出数据相同
        bias=0.5            #偏置
    )
    #x,y,coef此时为numpy对象
    print(type(x))
    print(type(y))
    print(type(coef))

    # 2.转换为tensor对象
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(y,dtype=torch.float)

    return x,y,coef

# 2.定义函数，表示模型训练
def train(x,y,coef):
    #2.1创建数据集对象，把tensor——>数据集对象——>数据加载器
    dataset = TensorDataset(x,y)

    #2.2创建数据加载器对象
    #参1：数据集对象，参2：批次大小，参3：是否打乱数据（训练集打乱，测试集不打乱）
    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    #2.3创建初始值，线性回归模型
    # 参1：输入特征维度(身高体重)，参2：输出特征维度(标签——人X).
    model = nn.Linear(1,1)

    #2.4创建损失函数对象
    criterion = nn.MSELoss()

    #2.5创建优化器对象
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    #2.6具体的训练过程
    #2.6.1 定义变量，分别表示：训练轮数，每轮的(平均)损失值，训练总损失值，训练的样本数.
    epochs,loss_list,total_loss,total_sample = 100,[],0.0,0
    #2.6.2 按轮训练
    for epoch in range(epochs):
        #2.6.3 每轮分批次，从数据加载器中获取
        for train_x,train_y in dataloader:
            #2.6.4 模型预测
            y_pred = model(train_x)
            #2.6.5 每批计算损失率
            loss = criterion(y_pred,train_y.reshape(-1,1))
            #2.6.6 计算总损失和样本批次数
            total_loss+=loss.item()
            total_sample+=1
            #2.6.7 梯度清零 + 反向传播 + 梯度更新.
            optimizer.zero_grad() #梯度清零
            loss.sum().backward() #反向传播
            optimizer.step()      #梯度更新
        #2.6.8把本轮的avg损失，添加到列表中
        loss_list.append(total_loss/total_sample)
        print(f'轮数:{epoch + 1}, 平均损失值: {total_loss / total_sample}')

    #2.7 打印最终训练结果
    print(f'{epochs} 轮的平均损失分别为: {loss_list}')
    print(f'模型参数，权重： {model.weight}，偏置： {model.bias}')

    #2.8 绘制损失曲线
    #2.8.1 100轮品均损失曲线
    plt.plot(range(epochs),loss_list)
    plt.title('损失值曲线')
    plt.grid(True)
    plt.show()

    #2.9 绘制预测值和真实值的关系.
    #2.9.1绘制样本点分布情况。
    plt.scatter(x,y)
    #2.9.2绘制训练模型的预测值,真实值
    y_pred = torch.tensor(data = [v*model.weight+model.bias for v in x])
    y_true = torch.tensor(data = [v*coef+0.5 for v in x])
    #绘图
    plt.plot(x, y_pred, color = 'red', label = '预测值')
    plt.plot(x, y_true, color = 'green', label = '真实值')
    plt.legend()
    plt.grid(True)
    plt.show()




# 3. 测试

if __name__=='__main__':
    #3.1创建数据集
    x,y,coef = create_dataset()

    #3.2训练模型
    train(x,y,coef)