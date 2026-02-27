# DeZero

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

一个轻量级的深度学习框架，从零开始实现自动求导和神经网络。DeZero致力于提供清晰、易懂的代码，帮助学习者深入理解深度学习的核心原理。

[English Version](./README_EN.md) | [中文版本](./README.md)

## 📋 目录

- [特性](#特性)
- [快速开始](#快速开始)
- [安装](#安装)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [核心概念](#核心概念)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

## ✨ 特性

- **自动求导（Autograd）**：自动计算梯度，支持反向传播算法
- **动态计算图**：灵活的计算图构建，支持控制流
- **神经网络模块**：提供常用的层和模型
- **CUDA支持**：支持GPU加速计算（可选）
- **易于理解**：代码注释详细，适合学习深度学习基础
- **兼容NumPy**：使用NumPy数组作为数据结构

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/DeZero.git
cd DeZero

# 安装依赖
pip install -r requirements.txt

# 或者直接安装
pip install -e .
```

### 基本使用

```python
import numpy as np
from dezero import Variable, Function

# 创建变量
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

# 定义计算
z = x ** 2 + y ** 3

# 反向传播
z.backward()

# 获取梯度
print(f"x的梯度: {x.grad}")
print(f"y的梯度: {y.grad}")
```

### 神经网络示例

```python
import numpy as np
from dezero import Variable, Model, Layer
import dezero.functions as F

# 定义模型
class TwoLayerNet(Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = Layer(input_size, hidden_size, activation=F.sigmoid)
        self.l2 = Layer(hidden_size, output_size)

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

# 创建模型
model = TwoLayerNet(10, 20, 1)

# 前向传播
x = Variable(np.random.randn(5, 10))
y = model(x)

# 反向传播
y.backward()
```

## 📦 安装

### 要求

- Python 3.8 或更高版本
- NumPy
- Matplotlib（可选，用于可视化）

### pip 安装

```bash
pip install dezero
```

### 从源代码安装

```bash
git clone https://github.com/yourusername/DeZero.git
cd DeZero
pip install -e .
```

## 💡 使用示例

### 1. 基本的自动求导

```python
from dezero import Variable
import numpy as np

# 创建变量
x = Variable(np.array(2.0))

# 定义函数：y = x^2
y = x ** 2

# 反向传播
y.backward()

# 打印梯度
print(x.grad)  # 4.0
```

### 2. 复杂计算图

```python
import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
a = x + x
b = a + x
y = b + 1

y.backward()
print(x.grad)  # 3.0
```

### 3. 使用函数库

```python
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))

# 使用内置函数
y = F.sum(F.exp(x))

y.backward()
print(x.grad)
```

### 4. 定义自己的函数

```python
from dezero import Function, Variable
import numpy as np

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

# 使用自定义函数
x = Variable(np.array(3.0))
y = square(x)
y.backward()
print(x.grad)  # 6.0
```

## 📁 项目结构

```
DeZero/
├── dezero/                 # 主包目录
│   ├── __init__.py        # 包初始化文件
│   ├── core.py            # 核心类：Variable, Function等
│   ├── core_simple.py     # 简化版本的核心实现
│   ├── functions.py       # 内置函数库
│   ├── layers.py          # 神经网络层
│   ├── models.py          # 模型基类
│   ├── optimizers.py      # 优化器
│   ├── datasets.py        # 数据集
│   ├── dataloaders.py     # 数据加载器
│   ├── transforms.py      # 数据变换
│   ├── cuda.py            # GPU支持
│   └── utils.py           # 工具函数
├── steps/                 # 学习步骤和教程
│   ├── steps01.py         # 基础概念
│   ├── steps02.py         # ...
│   └── ...
├── main.py               # 主程序示例
├── setup.py              # 安装配置
├── README.md             # 项目说明（中文）
├── LICENSE               # 许可证
├── .gitignore            # Git忽略文件
└── requirements.txt      # 项目依赖
```

## 🧠 核心概念

### Variable（变量）

`Variable`是框架的核心类，代表计算图中的节点，包含数据和梯度。

```python
from dezero import Variable
import numpy as np

x = Variable(np.array(2.0))
print(x.data)    # 获取数据
print(x.grad)    # 获取梯度（初始为None）
```

### Function（函数）

`Function`是计算操作的基类，定义前向传播和反向传播。

```python
from dezero import Function

class MyFunction(Function):
    def forward(self, x):
        # 前向传播计算
        return x ** 2
    
    def backward(self, gy):
        # 反向传播计算梯度
        x, = self.inputs
        return 2 * x * gy
```

### 反向传播

调用`backward()`方法自动计算所有变量的梯度。

```python
y.backward()  # 自动计算所有输入的梯度
print(x.grad) # 获取x的梯度
```

### 计算图

框架自动构建计算图，跟踪所有操作，用于梯度计算。

### 配置管理

使用`using_config`和`no_grad`管理计算行为。

```python
from dezero import using_config, no_grad

# 禁用反向传播
with no_grad():
    y = model(x)  # 不构建计算图

# 或者使用using_config
with using_config('enable_backprop', False):
    y = model(x)
```

## 🤝 贡献指南

我们欢迎所有形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细信息。

### 贡献步骤

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

### 报告问题

如果您发现任何问题，请通过 [GitHub Issues](https://github.com/yourusername/DeZero/issues) 报告。

## 📚 学习资源

- `steps/` 目录包含循序渐进的学习示例
- 每个步骤展示框架功能的不同方面
- 代码注释详细，适合初学者学习

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 👨‍💻 作者

- 项目维护者：[Your Name]

## 🙏 致谢

感谢所有贡献者和使用者的支持！

## 📞 联系方式

- GitHub Issues：[提交问题](https://github.com/yourusername/DeZero/issues)
- Email：[your.email@example.com]

---

**注意**：本项目主要用于教育和学习目的。如果您需要用于生产环境，请使用成熟的深度学习框架如 PyTorch 或 TensorFlow。
