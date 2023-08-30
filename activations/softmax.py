import numpy as np
import matplotlib.pylab as plt

def softmax (x) :
  exp = np.exp(x - np.max(x))
  exp_num = np.sum(exp)
  return exp / exp_num
