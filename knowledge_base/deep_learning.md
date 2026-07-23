# 深度学习基础：前馈网络与反向传播算法

## 1. 数学定义（通用近似框架）
深度前馈网络（Deep Feedforward Network）旨在逼近一个映射函数 $f^*$。定义输入 $\mathbf{x} \in \mathbb{R}^{d_0}$，输出 $\mathbf{y} \in \mathbb{R}^{d_L}$，网络通过 $L$ 层隐藏层的复合函数来定义：

$$f(\mathbf{x}; \Theta) = f^{(L)}(\mathbf{W}^{(L)} f^{(L-1)}(... f^{(1)}(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}) ... ) + \mathbf{b}^{(L)})$$

其中每一层的隐藏状态为 $\mathbf{h}^{(l)} = \phi(\mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$，$\phi$ 为非线性激活函数（如 ReLU: $\max(0, x)$）。

**通用近似定理（Universal Approximation Theorem）** （Hornik et al., 1991）：只需一个包含足够多神经元的隐藏层，前馈网络就能以任意精度逼近任意紧致子集上的连续函数。深度（多层）则是对指数级复杂函数的紧凑表达。

## 2. 关键方法论：反向传播（Backpropagation）算法
反向传播是深度学习的核心引擎，本质是**链式法则（Chain Rule）**在计算图中的高效实现。

- **前向传播**：计算每层输出 $\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}$，$\mathbf{h}^{(l)} = \phi(\mathbf{z}^{(l)})$。
- **损失函数**：定义标量损失 $L$（如负对数似然或交叉熵）。
- **误差反向传播**：定义 $\boldsymbol{\delta}^{(l)} = \frac{\partial L}{\partial \mathbf{z}^{(l)}}$ 为第 $l$ 层的误差项。
  - 输出层计算：$\boldsymbol{\delta}^{(L)} = 
abla_{\mathbf{z}} L \odot \phi'(\mathbf{z}^{(L)})$。

  - 隐藏层递推（梯度回传）：$\boldsymbol{\delta}^{(l)} = (\mathbf{W}^{(l+1)})^	op \boldsymbol{\delta}^{(l+1)} \odot \phi'(\mathbf{z}^{(l)})$。
- **参数梯度计算**：
  - $\frac{\partial L}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{h}^{(l-1)})^	op$
  - $\frac{\partial L}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$
- **优化**：使用基于梯度的随机梯度下降（SGD）及其变体（Adam、RMSprop），通过 $\Theta \leftarrow \Theta - \eta 
abla L$ 更新参数。

## 3. 研究演进脉络
- **1943年（萌芽）**：McCulloch & Pitts 提出 M-P 神经元模型。
- **1958年**：Rosenblatt 提出感知机（Perceptron），但被 Minsky 指出无法解决异或（XOR）问题，导致 AI 寒冬。
- **1986年**：Rumelhart, Hinton & Williams 重新发现并推广反向传播算法（此前由 Werbos 在 1974 年提出雏形），使多层网络训练成为可能。
- **2006年**：Hinton 提出深度信念网络（DBN），使用逐层贪心预训练解决深层网络难优化问题，正式确立“深度学习”之名。
- **2012至今**：引入 ReLU（修正线性单元）防止梯度饱和、Dropout 防止过拟合、Batch Normalization 平滑损失平面，使深度网络（ResNet等）可达数百层。


# 卷积神经网络：严格定义与谱视角研究

## 1. 数学定义（离散互相关与信号处理）
在深度学习实现中，CNN 的“卷积”实际为**互相关（Cross-correlation）**。令输入特征图 $\mathbf{X} \in \mathbb{R}^{H \times W \times C}$，卷积核 $\mathbf{K} \in \mathbb{R}^{k \times k \times C}$（通常采用 $C_{out}$ 个核），输出特征图在位置 $(i,j)$ 的值定义为：

$$\mathbf{Y}_{i,j} = \sum_{c=1}^{C} \sum_{u=1}^{k} \sum_{v=1}^{k} \mathbf{K}_{u,v,c} \cdot \mathbf{X}_{i+u-1, j+v-1, c} + b$$

其强归纳偏置（Inductive Bias）为：
1. **局部连接（Sparse Connectivity）**：核尺寸 $k \ll H, W$。
2. **权重共享（Parameter Sharing）**：同一核在空间维度滑动，参数数量与输入尺寸无关。
3. **平移等变性（Equivariance）**：$f(g \cdot \mathbf{X}) = g \cdot f(\mathbf{X})$，其中 $g$ 为平移变换。

## 2. 关键方法论：卷积层的反向传播（梯度转置卷积）
设 $\delta^{(l+1)}$ 为高层传回的误差，前向传播为 $\mathbf{Z}^{(l+1)} = \mathbf{X}^{(l)} \circledast \mathbf{K}$。

- **对卷积核的梯度**：$\frac{\partial L}{\partial \mathbf{K}} = \mathbf{X}^{(l)} \circledast \delta^{(l+1)}$（输入与误差的互相关）。
- **对输入（传给下层）的梯度**：$\delta^{(l)} = \delta^{(l+1)} \circledast \text{rot180}(\mathbf{K})$（将卷积核旋转 180 度后做完全卷积，等效于转置卷积操作）。
- **池化层（Pooling）**：作为下采样操作，其反向传播为“上采样（Upsampling）”。最大池化需记录前向传播中最大值的位置（索引），反向时将误差仅回传至该位置，其余位置填 0。

## 3. 研究演进脉络（空间维度与深度的博弈）
- **1989年（奠基）**：LeCun 提出 LeNet-5，使用 $5 \times 5$ 卷积和平均池化识别手写数字，确立 CNN 标准范本（Conv -> Pool -> Conv -> FC）。
- **2012年（复兴）**：AlexNet 在 ImageNet 夺冠。核心贡献：
  - 采用 ReLU 代替 $\tanh$，解决梯度饱和。
  - 引入 Dropout 正则化。
  - 使用 GPU 并行与数据增强（平移/镜像）。
- **2014-2015年（深度化）**：
  - **VGGNet**：证明网络性能随深度（16-19层）提升，使用连续小核 $3\times3$ 堆叠感受野等于大核。
  - **GoogLeNet (Inception)**：引入 $1\times1$ 卷积进行通道降维，并使用多尺度卷积核（$1\times1, 3\times3, 5\times5$）并行拼接。
- **2016年（残差革命）**：**ResNet**。定义残差模块 $\mathbf{y} = \mathcal{F}(\mathbf{x}, \{\mathbf{W}_i\}) + \mathbf{x}$。此恒等映射（Identity Mapping）使梯度可通过捷径（Shortcut）直接传回浅层，解决了深层退化问题，网络深度首次突破 1000 层。
- **后续**：DenseNet（密集连接特征重用）、MobileNet（深度可分离卷积降低计算量）、EfficientNet（神经架构搜索复合缩放）。


# 循环神经网络：时序动力学与长期依赖

## 1. 数学定义（状态空间模型）
RNN 是针对变长序列 $\{\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T\}$ 的确定性动态系统。其隐藏状态 $\mathbf{h}_t$ 通过非线性变换递归更新：

$$\mathbf{h}_t = \phi(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$$

$$\mathbf{y}_t = \psi(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

该定义的核心在于**时间步参数共享**（$\mathbf{W}_{xh}, \mathbf{W}_{hh}$ 在所有 $t$ 相同），使其能够泛化至任意序列长度。

## 2. 关键方法论：BPTT（时间反向传播）
训练 RNN 需将网络沿时间轴展开（Unroll），误差反向传播跨越时间步。

定义总损失 $L = \sum_{t=1}^{T} L_t(\mathbf{y}_t)$。对隐藏状态求导，根据链式法则，关键在于递推项 $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \mathbf{W}_{hh}^	op \cdot \text{diag}(\phi'(\mathbf{h}_{t+1}))$。

**梯度递推公式**：
$$\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L_t}{\partial \mathbf{h}_t} + \left( \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)^	op \frac{\partial L}{\partial \mathbf{h}_{t+1}}$$

对于权重 $\mathbf{W}_{hh}$ 的梯度为所有时间步贡献之和：
$$\frac{\partial L}{\partial \mathbf{W}_{hh}} = \sum_{k=1}^{T} \frac{\partial L_T}{\partial \mathbf{h}_T} \left( \prod_{i=k+1}^{T} \frac{\partial \mathbf{h}_i}{\partial \mathbf{h}_{i-1}} \right) \frac{\partial \mathbf{h}_k}{\partial \mathbf{W}_{hh}}$$

**本质困难（梯度消失/爆炸）**：连乘项 $\prod_{i=k+1}^{T} \text{diag}(\phi') \mathbf{W}_{hh}^	op$ 的特征值若大于 1 则爆炸（需梯度裁剪），若小于 1 则消失，导致无法捕获间隔超过 10 步的长依赖。

## 3. 变体与演进（门控机制解决长期依赖）

### 3.1 LSTM（长短期记忆网络，Hochreiter & Schmidhuber, 1997）
引入细胞状态（Cell State）$\mathbf{C}_t$ 作为信息高速公路。计算公式（遗忘门 $f_t$、输入门 $i_t$、输出门 $o_t$）：
- 遗忘门：$f_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_f)$
- 输入门：$i_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_i)$，候选值 $\tilde{\mathbf{C}}_t = \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_C)$
- 细胞状态更新：$\mathbf{C}_t = f_t \odot \mathbf{C}_{t-1} + i_t \odot \tilde{\mathbf{C}}_t$（此处 $f_t \approx 1$ 时梯度线性传递，解决消失）
- 输出门：$o_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_o)$，$\mathbf{h}_t = o_t \odot \tanh(\mathbf{C}_t)$

### 3.2 GRU（门控循环单元，Cho et al., 2014）
简化 LSTM，合并为两个门（更新门 $z_t$、重置门 $r_t$），参数更少，计算效率高。$\mathbf{h}_t = (1 - z_t) \odot \mathbf{h}_{t-1} + z_t \odot \tilde{\mathbf{h}}_t$。

- **2017年后（Transformer 时代）**：Transformer 基于纯注意力机制，允许并行计算且能捕捉极长依赖，在大规模语料上取代 RNN 成为 NLP 主流。但在流式语音、在线推理等需要隐式无限脉冲响应（IIR）滤波器的场景，RNN/LSTM 仍具不可替代性。
  


# 图神经网络通用框架：MPNN 与 WL 同构测试

## 1. 数学定义（MPNN 消息传递范式）
Gilmer et al. (2017) 统一了空间域 GNN，定义为 **消息传递神经网络（MPNN）**。给定图 $G=(V, E)$，节点特征 $\mathbf{h}_v^{(0)}$，边特征 $\mathbf{e}_{vw}$。经过 $K$ 步迭代，每步包含三个阶段：

- **消息计算（Message）**：$m_v^{(t+1)} = \bigoplus_{w \in \mathcal{N}(v)} M_t(\mathbf{h}_v^{(t)}, \mathbf{h}_w^{(t)}, \mathbf{e}_{vw})$，其中 $\bigoplus$ 是置换不变聚合函数（如 Sum, Mean, Max）。
- **节点更新（Update）**：$\mathbf{h}_v^{(t+1)} = U_t(\mathbf{h}_v^{(t)}, m_v^{(t+1)})$。
- **读出阶段（Readout）**：针对整图分类，$\hat{\mathbf{y}} = R(\{\mathbf{h}_v^{(K)} \mid v \in V\})$。

**核心约束（置换不变性）**：对于任意置换矩阵 $\mathbf{P}$，$f(\mathbf{P}\mathbf{X}) = f(\mathbf{X})$，即节点的输入顺序不影响输出。

## 2. 关键方法论：图神经网络的区分能力（WL 测试）

### 2.1 Weisfeiler-Lehman (WL) 同构测试
这是一种图同构的启发式算法。迭代地聚合邻域颜色（标签）并压缩为新的颜色：
$$\text{hash}(c_v^{(k)}, \{c_u^{(k)} \mid u \in \mathcal{N}(v)\}) \rightarrow c_v^{(k+1)}$$
若两图最终颜色分布不同，则必然非同构。

### 2.2 GIN（图同构网络，Xu et al., 2019）
为了达到 WL 测试的区分上限，**聚合函数必须为单射（Injective）**。作者理论证明，Sum 聚合器是单射的，而 Mean 和 Max 不是。GIN 的更新公式为：
$$\mathbf{h}_v^{(k)} = \text{MLP}^{(k)} \left( (1 + \epsilon^{(k)}) \odot \mathbf{h}_v^{(k-1)} + \sum_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(k-1)} \right)$$
其中 $\epsilon$ 是可学习或固定的标量，用于区分中心节点自身与邻居。

## 3. 研究演进脉络
- **2005-2009年（早期递归）**：Scarselli 等人提出早期 GNN，通过递归神经网络（RecGNN）迭代节点状态直到收敛（依赖 Banach 不动点定理），计算代价高且仅处理有向无环图。
- **2013年（谱域初探）**：Bruna 首次将谱图理论引入深度学习，但需对拉普拉斯矩阵进行特征分解（$\mathcal{O}(N^3)$），不适用于大规模图。
- **2017年（爆发元年）**：
  - **GCN**（Kipf & Welling）：将谱卷积简化为一阶近似。
  - **GraphSAGE**（Hamilton）：引入归纳学习（Inductive），通过采样固定大小邻居并使用聚合函数（Mean/LSTM/Pooling）扩展到大规模图。
  - **GAT**（Veličković）：引入自注意力机制，动态计算邻居权重 $\alpha_{vw}$，突破了固定权重的限制。
- **2019年（理论奠基）**：**GIN** 回答了“GNN 的上限是什么”，连接了 GNN 与图同构测试，为模型设计提供了理论指导。
- **近期趋势**：针对长程依赖（Long-range interactions）引入图 Transformer（如 Graphormer）；针对异构图和动态图的重构（如 HGCN、DySAT）。

# 图卷积网络：谱图理论的一阶近似实现

## 1. 数学定义（基于拉普拉斯矩阵的谱卷积）
GCN 严格推导自**谱图理论（Spectral Graph Theory）**。定义无向图 $G$，邻接矩阵 $\mathbf{A}$，度矩阵 $\mathbf{D}_{ii} = \sum_j \mathbf{A}_{ij}$。

### 1.1 归一化拉普拉斯矩阵
$$\mathbf{L} = \mathbf{I}_N - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$
因其对称半正定，可谱分解为 $\mathbf{L} = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^	op$，其中 $\mathbf{U}$ 是傅里叶基（特征向量），$\boldsymbol{\Lambda}$ 是特征值对角矩阵。

### 1.2 图傅里叶变换与卷积
信号 $\mathbf{x}$（节点特征）的图傅里叶变换为 $\hat{\mathbf{x}} = \mathbf{U}^	op \mathbf{x}$。卷积核 $\mathbf{g}_\theta$ 与信号 $\mathbf{x}$ 的卷积定义为傅里叶域的点积：
$$\mathbf{g}_\theta \star \mathbf{x} = \mathbf{U} \mathbf{g}_\theta(\boldsymbol{\Lambda}) \mathbf{U}^	op \mathbf{x}$$

## 2. 关键方法论：一阶切比雪夫近似（Chebyshev Expansion）
为了避开 $\mathbf{U}$ 的 $\mathcal{O}(N^3)$ 计算，Hammond et al. 提出使用切比雪夫多项式 $T_k(x)$ 近似 $\mathbf{g}_\theta(\boldsymbol{\Lambda})$：

$$\mathbf{g}_\theta(\boldsymbol{\Lambda}) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\boldsymbol{\Lambda}}) \quad \text{其中} \quad \tilde{\boldsymbol{\Lambda}} = 2\boldsymbol{\Lambda}/\lambda_{max} - \mathbf{I}$$

**Kipf & Welling (2017) 的两大核心简化**：
1. **限制卷积核阶数 $K=1$**（即只考虑一阶邻域，缓解过平滑）。
2. **令 $\lambda_{max} \approx 2$**，并固定 $\theta = \theta_0 = -\theta_1$ 以减少过拟合。

将近似代入并化简，最终得到简洁的传播规则：

### 2.3 重归一化技巧（Renormalization Trick）
直接使用 $\mathbf{I}_N + \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}$ 的谱半径可能大于 1，导致数值不稳定。因此向邻接矩阵添加自环：
$$\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N \quad , \quad \tilde{\mathbf{D}}_{ii} = \sum_j \tilde{\mathbf{A}}_{ij}$$
最终层间传播规则为：
$$\mathbf{H}^{(l+1)} = \sigma\left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^{(l)} \mathbf{W}^{(l)} \right)$$

**物理含义**：该公式等价于对每个节点及其邻居的特征进行**加权平均归一化的求和**（度数大的节点在聚合时贡献被稀释）。

## 3. 训练方法论与深层困境（过平滑）

- **训练模式**：GCN 采用**直推式学习（Transductive）**，即所有节点（包括测试集）均在训练时参与图结构构建，仅隐藏标签。损失函数通常为交叉熵，采用 Full-batch 梯度下降。
- **过平滑（Oversmoothing）**：当 $l \to \infty$ 时，多次应用拉普拉斯平滑算子会导致节点特征收敛至全局均值向量（子空间收敛）。经典 GCN 通常限制在 2-3 层。

## 4. 研究演进（针对 GCN 缺陷的修正）

- **深层 GCN 解法**：
  - **ResGCN / JK-Nets**：引入残差连接和跳跃知识网络，使浅层原始信息能直达深层。
  - **DropEdge** (Rong et al., 2020)：在每次训练迭代中随机删除部分边，增加邻域多样性，缓解过平滑并防止过拟合。
- **异配图（Heterophily）问题**：经典 GCN 假设相邻节点相似。针对异配（邻居类别不同），**Geom-GCN** 提出基于图几何关系的聚合，**H2GCN** 分离自身与邻居特征并融合高阶信息。
- **大规模扩展**：**Cluster-GCN** 使用图聚类算法将大图划分为密集子图，用于实施 Mini-batch SGD，解决了全图梯度下降难以承载工业级十亿节点图的问题。