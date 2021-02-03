# %%
# 主要展示pytorch的使用
import torch
import numpy as np

# %%

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# ndarray 转tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# %%
shape = (2, 3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# %%
# 张量的属性
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
# %%
# 张量索引和切片

tensor[:, 1] = 0  # 将第1列(从0开始)的数据全部赋值为0
print(tensor)
# %%
# torch
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
# %%
# 逐个元素相乘结果
print(f"tensor.mul(tensor): \n {tensor.mul(tensor)} \n")
# 等价写法:
print(f"tensor * tensor: \n {tensor * tensor}")
# %%
tensor1 = torch.tensor([[1, 2],
                        [2, 3],
                        [4, 5]])

# 转置矩阵
tensor2 = torch.tensor([[3, 2],
                        [5, 3],
                        [6, 5]]).T

# 矩阵乘法
print(f"tensor.matmul(tensor.T): \n {tensor1.matmul(tensor2)} \n")
# 等价写法:
print(f"tensor @ tensor.T: \n {tensor1 @ tensor2}")
