import numpy as np
from dezero import Variable
import dezero.functions as F

x0 = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
x1 = Variable(np.array(10))

y = x0 + x1
y.backward()
print(x0.grad)
