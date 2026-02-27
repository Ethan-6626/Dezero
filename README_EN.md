# DeZero

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A lightweight deep learning framework implementing automatic differentiation and neural networks from scratch. DeZero
aims to provide clear and understandable code to help learners deeply understand the core principles of deep learning.

[English Version](./README_EN.md) | [中文版本](./README.md)

## 📋 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Autograd**: Automatic gradient computation with support for backpropagation
- **Dynamic Computation Graph**: Flexible graph construction with control flow support
- **Neural Network Modules**: Common layers and models for neural networks
- **CUDA Support**: GPU acceleration support (optional)
- **Easy to Understand**: Well-commented code suitable for learning deep learning fundamentals
- **NumPy Compatible**: Uses NumPy arrays as the underlying data structure

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DeZero.git
cd DeZero

# Install dependencies
pip install -r requirements.txt

# Or install directly
pip install -e .
```

### Basic Usage

```python
import numpy as np
from dezero import Variable, Function

# Create variables
x = Variable(np.array(2.0))
y = Variable(np.array(3.0))

# Define computation
z = x ** 2 + y ** 3

# Backward propagation
z.backward()

# Get gradients
print(f"Gradient of x: {x.grad}")
print(f"Gradient of y: {y.grad}")
```

### Neural Network Example

```python
import numpy as np
from dezero import Variable, Model, Layer
import dezero.functions as F


# Define model
class TwoLayerNet(Model):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = Layer(input_size, hidden_size, activation=F.sigmoid)
        self.l2 = Layer(hidden_size, output_size)

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y


# Create model
model = TwoLayerNet(10, 20, 1)

# Forward propagation
x = Variable(np.random.randn(5, 10))
y = model(x)

# Backward propagation
y.backward()
```

## 📦 Installation

### Requirements

- Python 3.8 or higher
- NumPy
- Matplotlib (optional, for visualization)

### Using pip

```bash
pip install dezero
```

### From Source

```bash
git clone https://github.com/yourusername/DeZero.git
cd DeZero
pip install -e .
```

## 💡 Usage Examples

### 1. Basic Automatic Differentiation

```python
from dezero import Variable
import numpy as np

# Create variable
x = Variable(np.array(2.0))

# Define function: y = x^2
y = x ** 2

# Backward propagation
y.backward()

# Print gradient
print(x.grad)  # 4.0
```

### 2. Complex Computation Graph

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

### 3. Using Function Library

```python
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))

# Use built-in functions
y = F.sum(F.exp(x))

y.backward()
print(x.grad)
```

### 4. Defining Custom Functions

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


# Use custom function
x = Variable(np.array(3.0))
y = square(x)
y.backward()
print(x.grad)  # 6.0
```

### 5. Training a Simple Model

```python
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.models import MLP
from dezero import optimizers

# Generate sample data
np.random.seed(0)
x = Variable(np.random.randn(100, 10))
y = Variable(np.random.randint(0, 2, (100, 1)))

# Create model
model = MLP((10, 5, 1), activation=F.relu)
optimizer = optimizers.SGD(lr=0.01).setup(model)

# Training loop
for epoch in range(10):
    loss = F.softmax_cross_entropy(model(x), y)
    model.cleargrads()
    loss.backward()
    optimizer.update()
    print(f"Epoch {epoch + 1}: Loss = {float(loss.data):.4f}")
```

## 📁 Project Structure

```
DeZero/
├── dezero/                 # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── core.py            # Core classes: Variable, Function, etc.
│   ├── core_simple.py     # Simplified core implementation
│   ├── functions.py       # Built-in function library
│   ├── layers.py          # Neural network layers
│   ├── models.py          # Base model classes
│   ├── optimizers.py      # Optimizers (SGD, Adam, etc.)
│   ├── datasets.py        # Datasets (MNIST, CIFAR10, etc.)
│   ├── dataloaders.py     # Data loaders for batching
│   ├── transforms.py      # Data transformations
│   ├── cuda.py            # GPU support
│   └── utils.py           # Utility functions
├── steps/                 # Learning steps and tutorials
│   ├── steps01.py         # Basic concepts
│   ├── steps02.py         # ...
│   └── ...
├── main.py               # Main example program
├── setup.py              # Installation configuration
├── README_EN.md          # Project documentation (English)
├── README.md             # Project documentation (Chinese)
├── LICENSE               # License
├── .gitignore            # Git ignore file
└── requirements.txt      # Project dependencies
```

## 🧠 Core Concepts

### Variable

`Variable` is the core class of the framework, representing nodes in the computation graph, containing data and
gradients.

```python
from dezero import Variable
import numpy as np

x = Variable(np.array(2.0))
print(x.data)  # Get data
print(x.grad)  # Get gradient (initially None)
```

### Function

`Function` is the base class for computational operations, defining forward and backward propagation.

```python
from dezero import Function


class MyFunction(Function):
    def forward(self, x):
        # Forward computation
        return x ** 2

    def backward(self, gy):
        # Backward computation for gradients
        x, = self.inputs
        return 2 * x * gy
```

### Backward Propagation

Call the `backward()` method to automatically compute gradients for all variables.

```python
y.backward()  # Automatically compute gradients for all inputs
print(x.grad)  # Get gradient of x
```

### Computation Graph

The framework automatically builds a computation graph that tracks all operations for gradient computation.

### Configuration Management

Use `using_config` and `no_grad` to manage computational behavior.

```python
from dezero import using_config, no_grad

# Disable backward propagation
with no_grad():
    y = model(x)  # No computation graph is built

# Or use using_config
with using_config('enable_backprop', False):
    y = model(x)
```

### Optimizers

The framework provides various optimizers for training models.

```python
from dezero import optimizers

# Create optimizer
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

# Update model parameters
optimizer.update()
```

## 📚 Learning Resources

- The `steps/` directory contains step-by-step learning examples
- Each step demonstrates different aspects of the framework
- Code includes detailed comments suitable for beginners

## 🤝 Contributing

We welcome contributions of all kinds! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Contributing Steps

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Reporting Issues

If you find any issues, please report them through [GitHub Issues](https://github.com/yourusername/DeZero/issues).

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## 👨‍💻 Authors

- Project Maintainers: [Your Name]

## 🙏 Acknowledgments

Thanks to all contributors and users for their support!

## 📞 Contact

- GitHub Issues: [Submit Issues](https://github.com/yourusername/DeZero/issues)
- Email: [your.email@example.com]

---

**Note**: This project is primarily for educational and learning purposes. If you need it for production environments,
please use mature deep learning frameworks such as PyTorch or TensorFlow.

