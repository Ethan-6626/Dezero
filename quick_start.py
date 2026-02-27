#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速开始脚本 - 演示DeZero框架的基本用法
"""

import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dezero import Variable
import dezero.functions as F


def example_1_basic_autograd():
    """示例1：基本自动求导"""
    print("\n" + "=" * 60)
    print("示例1：基本自动求导")
    print("=" * 60)

    print("\n代码：")
    print("```python")
    print("x = Variable(np.array(2.0))")
    print("y = x ** 2")
    print("y.backward()")
    print("print(f'x.grad = {x.grad}')")
    print("```\n")

    x = Variable(np.array(2.0))
    y = x ** 2
    y.backward()
    print(f"结果：x.grad = {x.grad}")
    print("说明：y = x^2，所以 dy/dx = 2x = 4")


def example_2_complex_graph():
    """示例2：复杂计算图"""
    print("\n" + "=" * 60)
    print("示例2：复杂计算图")
    print("=" * 60)

    print("\n代码：")
    print("```python")
    print("x = Variable(np.array(2.0))")
    print("a = x + x")
    print("b = a + x")
    print("y = b + 1")
    print("y.backward()")
    print("print(f'x.grad = {x.grad}')")
    print("```\n")

    x = Variable(np.array(2.0))
    a = x + x
    b = a + x
    y = b + 1
    y.backward()
    print(f"结果：x.grad = {x.grad}")
    print("说明：y = (2x + x) + 1 = 3x + 1，所以 dy/dx = 3")


def example_3_functions():
    """示例3：使用内置函数"""
    print("\n" + "=" * 60)
    print("示例3：使用内置函数")
    print("=" * 60)

    print("\n代码：")
    print("```python")
    print("x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))")
    print("y = F.sum(F.exp(x))")
    print("y.backward()")
    print("print(f'y = {y}')")
    print("print(f'x.grad shape = {x.grad.shape}')")
    print("```\n")

    x = Variable(np.array([[1.0, 2.0], [3.0, 4.0]]))
    y = F.sum(F.exp(x))
    y.backward()
    print(f"结果：")
    print(f"  y = {y}")
    print(f"  x.grad.shape = {x.grad.shape}")
    print(f"  x.grad =\n{x.grad}")
    print("说明：对矩阵中的每个元素分别求导")


def example_4_custom_function():
    """示例4：定义自己的函数"""
    print("\n" + "=" * 60)
    print("示例4：定义自己的函数")
    print("=" * 60)

    print("\n代码：")
    print("```python")
    print("from dezero import Function")
    print("class Square(Function):")
    print("    def forward(self, x):")
    print("        return x ** 2")
    print("    def backward(self, gy):")
    print("        x, = self.inputs")
    print("        return 2 * x * gy")
    print("")
    print("def square(x):")
    print("    return Square()(x)")
    print("")
    print("x = Variable(np.array(3.0))")
    print("y = square(x)")
    print("y.backward()")
    print("print(f'x.grad = {x.grad}')")
    print("```\n")

    from dezero import Function

    class Square(Function):
        def forward(self, x):
            return x ** 2

        def backward(self, gy):
            x, = self.inputs
            return 2 * x * gy

    def square(x):
        return Square()(x)

    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    print(f"结果：x.grad = {x.grad}")
    print("说明：自定义Square函数，计算y = x^2的导数")


def example_5_no_grad():
    """示例5：禁用梯度计算"""
    print("\n" + "=" * 60)
    print("示例5：禁用梯度计算")
    print("=" * 60)

    print("\n代码：")
    print("```python")
    print("from dezero import no_grad")
    print("x = Variable(np.array(2.0))")
    print("")
    print("with no_grad():")
    print("    y = x ** 2")
    print("")
    print("print(f'y.creator = {y.creator}')")
    print("```\n")

    from dezero import no_grad
    x = Variable(np.array(2.0))

    with no_grad():
        y = x ** 2

    print(f"结果：y.creator = {y.creator}")
    print("说明：在no_grad()上下文中，不构建计算图")


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  DeZero 快速开始示例".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    print("\n这个脚本展示了DeZero框架的基本用法。\n")

    try:
        example_1_basic_autograd()
        example_2_complex_graph()
        example_3_functions()
        example_4_custom_function()
        example_5_no_grad()

        print("\n" + "=" * 60)
        print("✅ 所有示例执行成功！")
        print("=" * 60)
        print("\n更多示例请查看 steps/ 目录中的脚本文件。")
        print("完整文档请参阅 README.md\n")

        return True

    except Exception as e:
        print(f"\n❌ 执行失败：{e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
