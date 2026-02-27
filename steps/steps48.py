import dezero
import math
import numpy as np
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP
from dezero.datasets import get_spiral
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

max_epochs = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = get_spiral(train=True)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

# ==========================================
# 动画准备阶段：提前建好网格，准备一个列表存“快照”
# ==========================================
h = 0.05  # 为了让动画生成快一点，网格步长设为 0.05
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_grid = np.c_[xx.ravel(), yy.ravel()]

Z_history = []  # 用于保存每次预测的色块数据
epoch_history = []  # 用于保存对应的 Epoch 数字

print("开始训练并录制帧，请稍候...")

for epoch in range(max_epochs):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

    # ==========================================
    # 录制阶段：每 10 个 epoch 拍一张快照
    # ==========================================
    if (epoch + 1) % 10 == 0 or epoch == 0:
        with dezero.no_grad():
            score = model(X_grid)
        predict_cls = np.argmax(score.data, axis=1)
        Z = predict_cls.reshape(xx.shape)
        Z_history.append(Z)
        epoch_history.append(epoch + 1)

# ==========================================
# 播放阶段：将记录的快照逐帧渲染出来
# ==========================================
print("训练完成，正在生成动画...")
fig, ax = plt.subplots(figsize=(8, 6))
markers = ['o', 'x', '^']
colors = ['red', 'blue', 'green']


def update(frame_idx):
    ax.clear()  # 清空上一帧
    Z = Z_history[frame_idx]
    current_epoch = epoch_history[frame_idx]

    # 画当前的决策边界
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)

    # 画固定的数据点
    for i in range(3):
        mask = (t == i)
        ax.scatter(x[mask, 0], x[mask, 1], s=40, marker=markers[i], c=colors[i],
                   edgecolors='white' if markers[i] == 'o' else None, label=f'Class {i}')

    ax.set_title(f"Spiral Dataset Training - Epoch: {current_epoch}")
    ax.legend(loc='upper right')


# 创建动画对象 (interval=200 表示每帧间隔 200 毫秒)
ani = FuncAnimation(fig, update, frames=len(Z_history), interval=200, repeat=False)

plt.show()

