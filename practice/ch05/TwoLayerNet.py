import sys, os
sys.path.append(
  os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      '../../'
    )))

import numpy as np
from AffineLayer import AffineLayer
from ReluLayer import ReluLayer
from SoftmaxWithLossLayer import SoftmaxWithLossLayer
from practice.ch04.numerical_gradient import numerical_gradient

class TwoLayerNet :
  def __init__ (self, input_size, hidden_size, output_size, weight_init_std = 0.01) :
    self.params = {}
    self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
    self.params['b1'] = np.random.randn(hidden_size)
    self.params['W2'] = np.random.randn(hidden_size, output_size) * weight_init_std
    self.params['b2'] = np.random.randn(output_size)

    self.layers = []
    self.layers.append(AffineLayer(self.params['W1'], self.params['b1']))
    self.layers.append(ReluLayer())
    self.layers.append(AffineLayer(self.params['W2'], self.params['b2']))
    self.lastLayer = SoftmaxWithLossLayer()

  def predict (self, x) :
    for layer in self.layers :
      x = layer.forward(x)
    return x

  def loss (self, x, t) :
    y = self.predict(x)
    loss = self.lastLayer.forward(y, t)
    return loss

  def accuracy (self, x, t) :
    y = self.predict(x)
    y = np.argmax(y, axis = 1)
    if t.ndim != 1 :
      t = np.argmax(t, axis = 1)

    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  def numerical_gradient (self, x, t) :
    loss_W = lambda w: self.loss(x, t)

    grad = {}
    grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
    grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
    grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
    grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
    return grad

  def gradient (self, x, t) :
    self.loss(x, t)

    dOut = 1.0
    dOut = self.lastLayer.backward(dOut)

    self.layers.reverse()

    for layer in self.layers :
      dOut = layer.backward(dOut)

    self.layers.reverse()

    grad = {}
    grad['W1'] = self.layers[0].dW
    grad['b1'] = self.layers[0].db
    grad['W2'] = self.layers[2].dW
    grad['b2'] = self.layers[2].db
    return grad
