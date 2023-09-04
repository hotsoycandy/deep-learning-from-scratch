import numpy as np

class AffineLayer :
  def __init__ (self, W, b) :
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward (self, x) :
    self.x = x
    return np.dot(self.x, self.W) + self.b

  def backward (self, dOut) :
    dx = np.dot(dOut, self.W.T)
    self.dW = np.dot(self.x.T, dOut)
    self.db = np.sum(dOut, axis = 0)
    return dx
