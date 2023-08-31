import numpy as np

def cross_entropy_error (y, t) :
  if y.ndim == 1 :
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
  if t.size == y.size :
    t = t.argmax(axis=1)

  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def cross_entropy_error_old (y, t) :
  delta = 1e-7 # delta for preventing to call np.log with 0
  return -np.sum(t * np.log(y + delta))

if __name__ == '__main__' :
  t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
  y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
  print(cross_entropy_error(y, t))

  y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
  print(cross_entropy_error(y, t))
