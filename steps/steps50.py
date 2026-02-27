import dezero
import numpy as np
import matplotlib.pyplot as plt  # 新增：画图库
from dezero.datasets import Spiral
from dezero import DataLoader, optimizers
from dezero.models import MLP
import dezero.functions as F

max_epochs = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

# 新增：准备 4 个空列表，用来当"小本本"，记录每一个 epoch 的成绩
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

for epoch in range(max_epochs):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc) * len(t)

    # 计算平均分
    avg_train_loss = sum_loss / len(train_set)
    avg_train_acc = sum_acc / len(train_set)
    print('epoch: {}'.format(epoch + 1))
    print('train_loss:{:.4f}, accuracy:{:.4f}'.format(avg_train_loss, avg_train_acc))

    # 新增：记入小本本
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc) * len(t)

    # 计算平均分
    avg_test_loss = sum_loss / len(test_set)
    avg_test_acc = sum_acc / len(test_set)
    print('test_loss:{:.4f}, accuracy:{:.4f}'.format(avg_test_loss, avg_test_acc))

    # 新增：记入小本本
    test_loss_list.append(avg_test_loss)
    test_acc_list.append(avg_test_acc)

# ==========================================
# 训练结束，开始全场最酷炫的可视化环节！
# ==========================================

# 1. 画出 Loss 和 Accuracy 曲线
plt.figure(figsize=(12, 5))

# 1.1 画 Loss 子图
plt.subplot(1, 2, 1)
plt.plot(train_loss_list, label='train', color='blue')
plt.plot(test_loss_list, label='test', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# 1.2 画 Accuracy 子图
plt.subplot(1, 2, 2)
plt.plot(train_acc_list, label='train', color='blue')
plt.plot(test_acc_list, label='test', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()

# 2. 画出决策边界 (Decision Boundary)
# 生成网格点，覆盖整个二维平面
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]  # 把网格变成模型认识的二维坐标集 (N, 2)

# 让模型来预测整个平面的得分
with dezero.no_grad():
    score = model(X.astype(np.float32))
    predict_cls = np.argmax(score.data, axis=1)  # 选取得分最高的类别

# 把预测结果变回网格的形状并画图
Z = predict_cls.reshape(xx.shape)
plt.figure(figsize=(8, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # 画出不同颜色的地盘

# 把原始的训练数据点也画上去，看模型包围得准不准
train_x, train_t = train_set.data, train_set.label
# 将 3 个类别用不同的颜色标出来
colors = ['red', 'blue', 'green']
markers = ['o', 'x', '^']
for i in range(3):
    mask = (train_t == i)
    plt.scatter(train_x[mask, 0], train_x[mask, 1], color=colors[i], marker=markers[i], label=f'Class {i}')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.title("Decision Boundary of MLP")
plt.legend()
plt.show()
