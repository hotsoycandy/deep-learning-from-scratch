import numpy as np
import matplotlib.pylab as plt

def step (x) :
  y = x > 0
  print(y.astype(np.int32))
  return y.astype(np.int32)

if __name__ == '__main__' :
  x = np.arange(-5, 5, 0.1)
  y = step(x)

  plt.plot(x, y)
  plt.show()
