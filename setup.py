from setuptools import setup, find_packages

# 读取README用于长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='dezero',
    version='0.1.0',
    author='DeZero Contributors',
    description='一个轻量级的深度学习框架，从零开始实现自动求导和神经网络',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yourusername/DeZero',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=install_requires,
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=2.10.0',
            'pylint>=2.6.0',
            'flake8>=3.8.0',
            'yapf>=0.30.0',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/DeZero/issues',
        'Source': 'https://github.com/yourusername/DeZero',
        'Documentation': 'https://github.com/yourusername/DeZero/blob/main/README.md',
    },
)
