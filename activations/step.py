import numpy as np
import matplotlib.pylab as plt

def step (x) :
  y = x > 0
  print(y.astype(np.int32))
  return y.astype(np.int32)
