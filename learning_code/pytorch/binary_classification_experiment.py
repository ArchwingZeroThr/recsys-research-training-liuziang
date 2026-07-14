# 0. 导入依赖并固定随机性
from copy import deepcopy
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("PyTorch版本：", torch.__version__)
print("随机种子：", SEED)


# 1. 生成二维二分类数据

# 设置每个类别的样本数量。
# 本例包含两个类别，因此最终会生成 120 × 2 = 240 个样本。
num_per_class = 120

# 创建一个独立的随机数生成器，并使用 SEED 设置随机种子。
# 固定随机种子可以保证每次运行代码时生成相同的随机数据，便于实验复现。
generator = torch.Generator().manual_seed(SEED)

# 生成类别 0 的样本：
# 1. torch.randn(num_per_class, 2) 生成 120 个二维标准正态分布样本；
# 2. 乘以 0.70，控制样本的离散程度；
# 3. 加上 [-1.0, -1.0]，将样本中心移动到坐标 (-1, -1) 附近。
class_0 = (
    torch.randn(num_per_class, 2, generator=generator) * 0.70
    + torch.tensor([-1.0, -1.0])
)

# 生成类别 1 的样本。
# 生成方式与类别 0 相同，但样本中心被移动到坐标 (1, 1) 附近。
class_1 = (
    torch.randn(num_per_class, 2, generator=generator) * 0.70
    + torch.tensor([1.0, 1.0])
)

# 沿第 0 维（样本维度）拼接两个类别的特征。
# 拼接后的 features 形状为 [240, 2]：
# - 240 表示样本总数；
# - 2 表示每个样本有两个特征。
# float() 将数据类型统一转换为 torch.float32。
features = torch.cat([class_0, class_1], dim=0).float()

# 为两个类别创建标签：
# - 类别 0 的 120 个样本对应标签 0；
# - 类别 1 的 120 个样本对应标签 1。
# 拼接后的 labels 形状为 [240]。
# 二分类模型通常使用浮点型标签计算二元交叉熵损失。
labels = torch.cat(
    [
        torch.zeros(num_per_class),
        torch.ones(num_per_class),
    ],
    dim=0,
).float()

# 输出特征张量的形状和数据类型，检查数据是否符合预期。
print("features:", features.shape, features.dtype)

# 输出标签张量的形状和数据类型。
print("labels:  ", labels.shape, labels.dtype)
# 2. 画图：先观察数据，再写模型
plt.figure(figsize=(7, 5))
plt.scatter(features[labels == 0, 0], features[labels == 0, 1], alpha=0.7, label="label 0")
plt.scatter(features[labels == 1, 0], features[labels == 1, 1], alpha=0.7, label="label 1")
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Synthetic 2D Binary Classification Data")
plt.legend(); plt.grid(alpha=0.2); plt.show()



# 3. 固定随机划分train / validation / test
indices = torch.randperm(len(features), generator=torch.Generator().manual_seed(SEED))
train_idx, valid_idx, test_idx = indices[:160], indices[160:200], indices[200:]

train_x, train_y = features[train_idx], labels[train_idx]
valid_x, valid_y = features[valid_idx], labels[valid_idx]
test_x, test_y = features[test_idx], labels[test_idx]
print("train / valid / test：", len(train_x), len(valid_x), len(test_x))


# 4. Dataset → DataLoader → batch
BATCH_SIZE = 16
train_set = TensorDataset(train_x, train_y)
valid_set = TensorDataset(valid_x, valid_y)
test_set = TensorDataset(test_x, test_y)

train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
valid_loader = DataLoader(valid_set, BATCH_SIZE)
test_loader = DataLoader(test_set, BATCH_SIZE)
print("训练集batch数量：", len(train_loader))


# 5. 查看一个batch
preview_loader = DataLoader(train_set, BATCH_SIZE, shuffle=False)  # 仅用于课堂预览，不影响训练顺序
batch_features, batch_labels = next(iter(preview_loader))
print("batch_features.shape：", batch_features.shape)
print("batch_labels.shape：  ", batch_labels.shape)
print("features dtype：", batch_features.dtype)
print("labels dtype：  ", batch_labels.dtype)
print("第一条样本：", batch_features[0], "label=", batch_labels[0])


# 6. 定义最小模型：[B, 2] → [B, 1] → [B]
model = nn.Linear(in_features=2, out_features=1)
with torch.no_grad():
    model.weight.copy_(torch.tensor([[0.4, -0.2]]))
    model.bias.fill_(0.1)

logits = model(batch_features).squeeze(dim=1)
print(model)
print("logits.shape：", logits.shape)
print("labels.shape：", batch_labels.shape)


# 7. 一个batch的loss必须是标量
loss_fn = nn.BCEWithLogitsLoss()
logits = model(batch_features).squeeze(dim=1)
loss = loss_fn(logits, batch_labels)
print("logits.shape：", logits.shape)
print("labels.shape：", batch_labels.shape)
print("loss.shape：", loss.shape)
print("loss：", loss.item())


# 8. 执行一次更新，确认参数真的改变
update_demo = deepcopy(model)
optimizer = torch.optim.SGD(update_demo.parameters(), lr=0.2)
before = update_demo.weight.detach().clone()

optimizer.zero_grad()
demo_loss = loss_fn(update_demo(batch_features).squeeze(1), batch_labels)
demo_loss.backward()
optimizer.step()

after = update_demo.weight.detach().clone()
print("更新前：", before)
print("更新后：", after)
print("参数变化量：", (after-before).abs().sum().item())


# 9. 验证函数：只评价，不更新参数
def evaluate(model, data_loader):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x).squeeze(1)
            total_loss += loss_fn(logits, y).item() * len(y)
            predictions = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (predictions == y).sum().item()
            total_count += len(y)
    return total_loss / total_count, total_correct / total_count


# 10. 一个epoch：训练集参与参数更新
def train_one_epoch(model, data_loader, optimizer):
    model.train()
    total_loss, total_count = 0.0, 0
    for x, y in data_loader:
        optimizer.zero_grad()
        logits = model(x).squeeze(1)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        total_count += len(y)
    return total_loss / total_count


# 11. 完整训练：验证集选择最佳模型
model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
history, best_state, best_valid_loss = [], None, float("inf")

for epoch in range(1, 31):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    valid_loss, valid_acc = evaluate(model, valid_loader)
    history.append({"epoch": epoch, "train_loss": train_loss,
                    "valid_loss": valid_loss, "valid_acc": valid_acc})
    if valid_loss < best_valid_loss:
        best_valid_loss, best_state = valid_loss, deepcopy(model.state_dict())
    if epoch == 1 or epoch % 5 == 0:
        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | "
              f"valid {valid_loss:.4f} | acc {valid_acc:.3f}")

history = pd.DataFrame(history)

# 12. 画出训练与验证loss
plt.figure(figsize=(7, 4.5))
plt.plot(history["epoch"], history["train_loss"], label="train loss")
plt.plot(history["epoch"], history["valid_loss"], label="validation loss")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("Training and Validation Loss")
plt.legend(); plt.grid(alpha=0.2); plt.show()

history.tail()


# 13. 恢复最佳验证模型，最后检查test集
best_epoch = int(history.loc[history["valid_loss"].idxmin(), "epoch"])
model.load_state_dict(best_state)
test_loss, test_acc = evaluate(model, test_loader)

print("最佳验证epoch：", best_epoch)
print("最佳验证loss：", round(best_valid_loss, 4))
print("最终test loss：", round(test_loss, 4))
print("最终test accuracy：", round(test_acc, 3))


# 14. 用shape表复述完整数据流
shape_table = pd.DataFrame([
    ["全部输入", "features", tuple(features.shape)],
    ["一个batch", "batch_features", tuple(batch_features.shape)],
    ["模型输出", "logits", tuple(model(batch_features).squeeze(1).shape)],
    ["真实标签", "batch_labels", tuple(batch_labels.shape)],
    ["批次损失", "loss", tuple(loss.shape)],
], columns=["阶段", "变量", "shape"])
shape_table
