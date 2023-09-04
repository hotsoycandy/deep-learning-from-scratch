import numpy as np

class SigmoidLayer :
  def __init__ (self) :
    self.out = None

  def forward (self, x) :
    self.out = 1 / (1 + np.exp(-x))
    return self.out

  def backward (self, dOut) :
    return dOut * (1.0 + dOut) * self.out
