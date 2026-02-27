from dezero import Variable, np
import dezero.functions as F
import matplotlib.pyplot as plt

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))


def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid_simple(y)
    y = F.linear(y, W2, b2)
    return y


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_squared_error(y_pred, y)

    W1.cleargrad()
    b1.cleargrad()
    W2.cleargrad()
    b2.cleargrad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(loss)

# ==========================================
# 训练结束，开始可视化展示！
# ==========================================

# 1. 画出原始的散点数据（蓝点）
plt.scatter(x, y, s=10, label='y = sin(2*pi*x) + noise')

# 2. 生成密集且连续的 x 坐标，用来画平滑的预测曲线
# np.arange(0, 1, 0.01) 会生成 [0.0, 0.01, 0.02, ... 0.99]
# [:, np.newaxis] 是为了把它变成形状为 (100, 1) 的二维矩阵，和训练数据保持一致
t = np.arange(0, 1, 0.01)[:, np.newaxis]

# 3. 让训练好的神经网络对这些连续的 t 进行预测
y_pred = predict(t)

# 4. 画出神经网络预测的红色曲线
# 注意：y_pred 是一个 Variable 对象，画图时必须加上 .data 剥离出纯 Numpy 数组
plt.plot(t, y_pred.data, color='r', label='Neural Network Predict')

# 5. 添加图例和标签，并显示图像
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
