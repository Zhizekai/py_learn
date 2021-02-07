# %%
# 展示神经网络的使用
import numpy as np
import torch, torchvision
# %%
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64) # size(1,3,64,64) 一张 三通道 像素64 * 64
labels = torch.rand(1, 1000)
# %%
prediction = model(data) # forward pass

# %%
loss = (prediction - labels).sum()
loss.backward() # backward pass
# %%
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
optim.step() #gradient descent
# %%
# 解释autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
# %%
Q = 3*a**3 - b**2

external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad) # 向后传播，计算a和b的偏导
# %%
# check if collected gradients are correct
print(9*a**2 == a.grad) # 查看a的梯度，就是Q对a求偏导
print(-2*b == b.grad)

