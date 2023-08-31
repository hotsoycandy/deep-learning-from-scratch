import numpy as np

def cross_entropy_error (y, t) :
  delta = 1e-7 # delta for preventing to call np.log with 0
  return -np.sum(t * np.log(y + delta))

if __name__ == '__main__' :
  t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
  y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
  print(cross_entropy_error(y, t))

  y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
  print(cross_entropy_error(y, t))
