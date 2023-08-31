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

class TwoLayerNet :
  def __init__ (self, input_size, hidden_size, output_size, weight_init_std = 00.1) :
    self.params = {}
    self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def predict (self, x) :
    a1 = np.dot(x, self.params['W1']) + self.params['b1']
    z1 = sigmoid(a1)
    a2 = np.dot(z1, self.params['W2']) + self.params['b2']
    y = softmax(a2)
    return y

  def loss (self, x, t) :
    y = self.predict(x)
    return cross_entropy_error(y, t)

  def accuracy (self, x, t) :
    y = self.predict(x)
    y = np.argmax(y, axis = 1)
    t = np.argmax(t, axis = 1)

    return np.sum(y == t) / float(x.shape[0])

  def numerical_gradient (self, x, t) :
    loss_W = lambda w: self.loss(x, t)

    grad = {}
    grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
    return grad

if __name__ == '__main__' :
  net = TwoLayerNet(input_size = 784, hidden_size = 100, output_size = 10)
  print(net.params['W1'].shape)
  print(net.params['b1'].shape)
  print(net.params['W2'].shape)
  print(net.params['b2'].shape)

  x = np.random.randn(100, 784)
  y = net.predict(x)
  print(y.shape)

  t = np.random.rand(100, 10)
  grad = net.numerical_gradient(x, y)
  print(grad['W1'].shape)
  print(grad['b1'].shape)
  print(grad['W2'].shape)
  print(grad['b2'].shape)
