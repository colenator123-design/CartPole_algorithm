import numpy as np
import torch
from matplotlib import pyplot as plt

# 設置運行設備為 CPU，避免 CUDA 警告
device = torch.device("cpu")

# 初始化自旋格子
def init_grid(size=(10,)):
    grid = torch.randn(*size).to(device)
    grid[grid > 0] = 1
    grid[grid <= 0] = 0
    grid = grid.byte()
    return grid

# 定義獎勵函數
def get_reward(s, a):
    r = -1
    for i in s:
        if i == a:
            r += 0.9
    r *= 2.
    return r

# 生成參數
def gen_params(N, size):
    ret = []
    for i in range(N):
        vec = torch.randn(size).to(device) / 10.
        vec.requires_grad = True
        ret.append(vec)
    return ret

# Q 函數
def qfunc(s, theta, layers=[(4, 20), (20, 2)], afn=torch.tanh):
    l1n = layers[0]
    l1s = np.prod(l1n)
    theta_1 = theta[0:l1s].reshape(l1n)
    l2n = layers[1]
    l2s = np.prod(l2n)
    theta_2 = theta[l1s:l2s+l1s].reshape(l2n)
    bias = torch.ones((1, theta_1.shape[1])).to(device)
    l1 = s @ theta_1 + bias
    l1 = torch.nn.functional.elu(l1)
    l2 = afn(l1 @ theta_2)
    return l2.flatten()

# 獲取單個自旋狀態
def get_substate(b):
    s = torch.zeros(2).to(device)
    if b > 0:
        s[1] = 1
    else:
        s[0] = 1
    return s

# 聯合狀態
def joint_state(s):
    s1_ = get_substate(s[0])
    s2_ = get_substate(s[1])
    ret = (s1_.reshape(2,1) @ s2_.reshape(1,2)).flatten()
    return ret

# 訓練及視覺化
plt.figure(figsize=(8, 5))
size = (20,)
hid_layer = 20
params = gen_params(size[0], 4 * hid_layer + hid_layer * 2)
grid = init_grid(size=size)
grid_ = grid.clone()
print(grid)

# 保存初始狀態圖片
plt.imshow(np.expand_dims(grid, 0))
plt.savefig('initial_grid.png')

epochs = 200
lr = 0.001
losses = [[] for i in range(size[0])]

# 訓練過程
for i in range(epochs):
    for j in range(size[0]):
        l = j - 1 if j - 1 >= 0 else size[0] - 1
        r = j + 1 if j + 1 < size[0] else 0
        state_ = grid[[l, r]]
        state = joint_state(state_)
        qvals = qfunc(state.float().detach(), params[j], layers=[(4, hid_layer), (hid_layer, 2)])
        qmax = torch.argmax(qvals, dim=0).detach().item()
        action = int(qmax)
        grid_[j] = action
        reward = get_reward(state_.detach(), action)
        with torch.no_grad():
            target = qvals.clone()
            target[action] = reward
        loss = torch.sum(torch.pow(qvals - target, 2))
        losses[j].append(loss.detach().numpy())
        loss.backward()
        with torch.no_grad():
            params[j] = params[j] - lr * params[j].grad
        params[j].requires_grad = True
    with torch.no_grad():
        grid.data = grid_.data

# 視覺化損失和最終結果
fig, ax = plt.subplots(2, 1)
for i in range(size[0]):
    ax[0].scatter(np.arange(len(losses[i])), losses[i])

print(grid, grid.sum())

# 保存最終格子狀態圖片
ax[1].imshow(np.expand_dims(grid, 0))
plt.savefig('final_grid.png')
