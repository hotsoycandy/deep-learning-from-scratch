class ReluLayer :
  def __init__ (self) :
    self.mask = None

  def forward (self, x) :
    self.mask = x <= 0
    out = x.copy()
    out[self.mask] = 0
    return out

  def backward (self, dOut) :
    dOut[self.mask] = 0
    return dOut
