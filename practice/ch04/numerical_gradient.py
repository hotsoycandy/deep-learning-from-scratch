import numpy as np

def numerical_gradient (f, x) :
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
      idx = it.multi_index
      tmp_val = x[idx]
      x[idx] = float(tmp_val) + h
      fxh1 = f(x) # f(x+h)

      x[idx] = tmp_val - h
      fxh2 = f(x) # f(x-h)
      grad[idx] = (fxh1 - fxh2) / (2*h)

      x[idx] = tmp_val # 값 복원
      it.iternext()

  return grad

def numerical_gradient_old (f, x) :
  h = 1e-4
  grad = np.zeros_like(x)

  for i in range(x.size) :
    origianl_x = x[i]

    # f(x + h)
    x[i] = origianl_x + h
    fxh1 = f(x)
    # f(x - h)
    x[i] = origianl_x - h
    fxh2 = f(x)

    grad[i] = (fxh1 - fxh2) / (h * 2)

    x[i] = origianl_x

  return grad

if __name__ == '__main__' :
  def function1 (x) :
    return x[0] ** 2 + x[1] ** 2

  res1 = numerical_gradient(function1, np.array([3.0, 4.0]))
  print(res1)
  res2 = numerical_gradient(function1, np.array([0.0, 2.0]))
  print(res2)
  res3 = numerical_gradient(function1, np.array([3.0, 0.0]))
  print(res3)
