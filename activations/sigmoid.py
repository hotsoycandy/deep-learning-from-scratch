import numpy as np
import matplotlib.pylab as plt

def sigmoid (x) :
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__' :
  x = np.arange(-5, 5, 0.1)
  y = sigmoid(x)

  plt.plot(x, y)
  plt.xlim(-5.1, 5.1)
  plt.ylim(-0.1, 1.1)
  plt.show()
