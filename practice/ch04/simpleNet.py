# configure to import internal modules
import sys, os
sys.path.append(
  os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      '../../'
    )))

import numpy as np
from activations.sigmoid import sigmoid
from activations.softmax import softmax
from losses.cross_entropy_error import cross_entropy_error
from numerical_gradient import numerical_gradient

class simpleNet :
  def __init__ (self) :
    self.W = np.random.randn(2, 3)

  def predict (self, x) :
    return softmax(np.dot(x, self.W))

  def loss (self, x ,t) :
    z = self.predict(x)
    y = sigmoid(z)
    loss = cross_entropy_error(y, t)

    return loss

if __name__ == '__main__' :
  net = simpleNet()
  print(net.W)

  x = np.array([0.6, 0.9])
  p = net.predict(x)
  print(p, np.argmax(p))

  t = np.array([0, 0, 1])
  print(net.loss(x, t))

  dw = numerical_gradient(
    lambda w: net.loss(x, t),
    net.W
  )
  print(dw)
