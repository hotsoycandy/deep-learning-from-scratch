import sys, os
sys.path.append(
  os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      '../../'
    )))

import numpy as np
from activations.softmax import softmax
from losses.cross_entropy_error import cross_entropy_error

class SoftmaxWithLossLayer :
  def __init__ (self) :
    self.y = None
    self.t = None

  def forward (self, x, t) :
    # exp = np.exp(x - np.max(x))
    # exp_sum = np.sum(exp)
    # y = exp / exp_sum
    self.t = t
    self.y = softmax(x)
    return cross_entropy_error(self.y, t)

  def backward (self, dOut = 1) :
    batch_size = self.t.shape[0]
    return (self.y - self.t) / batch_size
