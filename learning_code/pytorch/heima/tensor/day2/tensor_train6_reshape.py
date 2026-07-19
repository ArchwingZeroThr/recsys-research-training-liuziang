import torch
# 准备一个顺序张量便于观察
t = torch.arange(1, 13)   # 1 到 12
print(t)
# Output: tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# ----- 1. reshape / view -----
# 变换为 (3, 4)
t_reshaped = t.reshape(3, 4)
print(t_reshaped)
# Output: tensor([[ 1,  2,  3,  4],
#                 [ 5,  6,  7,  8],
#                 [ 9, 10, 11, 12]])

# view 要求内存连续，reshape 更通用
t_view = t.view(2, 6)
print(t_view)
# Output: tensor([[ 1,  2,  3,  4,  5,  6],
#                 [ 7,  8,  9, 10, 11, 12]])

# ----- 2. 展平（Flatten）-----
t_flat = t_reshaped.flatten()
print(t_flat)
# Output: tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# 等价写法
t_flat2 = t_reshaped.reshape(-1)
print(t_flat2)
# Output: tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# ----- 3. 转置（二维）-----
t_matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t_matrix.t())
# Output: tensor([[1, 4],
#                 [2, 5],
#                 [3, 6]])

# ----- 4. 增加/删除维度 -----
t = torch.randint(1, 10, (4, 3))
print(t.shape)        # Output: torch.Size([4, 3])

# 在索引1处插入维度（变为 4x1x3）
t_unsqueeze = t.unsqueeze(1)
print(t_unsqueeze.shape)
# Output: torch.Size([4, 1, 3])

# 删除维度为1的维度（变回 4x3）
t_squeeze = t_unsqueeze.squeeze(1)
print(t_squeeze.shape)
# Output: torch.Size([4, 3])

# ----- 5. 维度交换（permute）-----
t3d = torch.randint(1, 10, (2, 3, 4))
print(t3d.shape)      # Output: torch.Size([2, 3, 4])

# 将维度顺序从 (0,1,2) 交换为 (2,0,1)
t_permuted = t3d.permute(2, 0, 1)
print(t_permuted.shape)
# Output: torch.Size([4, 2, 3])