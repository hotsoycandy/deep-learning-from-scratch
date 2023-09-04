import sys, os
sys.path.append(
  os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      '../../'
    )))

import numpy as np
from TwoLayerNet import TwoLayerNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) \
  = load_mnist(
    normalize = True,
    one_hot_label = True
  )

network = TwoLayerNet(784, 50, 10)
x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in ('W1', 'b1', 'W2', 'b2') :
  diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
  print(key + ':' + str(diff))
