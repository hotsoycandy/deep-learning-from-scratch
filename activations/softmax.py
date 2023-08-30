import numpy as np
import matplotlib.pylab as plt

def softmax (x) :
  exp = np.exp(x - np.max(x))
  exp_num = np.sum(exp)
  return exp / exp_num

if __name__ == '__main__' :
  x = np.array([4, 7, 6, 3])
  y = softmax(x)

  plt.bar(x, y)
  plt.show()
