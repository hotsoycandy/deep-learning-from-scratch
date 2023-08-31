import numpy as np
from numerical_gradient import numerical_gradient

def gradient_descent (f, init_x, lr = 0.01, step_num = 100) :
  x = init_x

  for _ in range(step_num) :
    grad = numerical_gradient(f, x)
    x -= grad * lr

  return x

if __name__ == '__main__' :
  def function1 (x) :
    return x[0] ** 2 + x[1] ** 2

  init_x = np.array([-3.0, 4.0])

  result = gradient_descent(function1, init_x, lr = 0.1)
  print (result)
