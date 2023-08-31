import sys, os
sys.path.append(
  os.path.abspath(
    os.path.join(
      os.path.dirname(__file__),
      '../../'
    )))

import numpy as np
from matplotlib.pylab import plt
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet

(x_train, t_train), (x_test, t_test) \
  = load_mnist(
    normalize = True,
    one_hot_label = True
  )

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 하이퍼파라메터
iter_sum = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_late = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

net = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

for i in range(iter_sum) :
  print('epoch: ', i)
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  grad = net.numerical_gradient(x_batch, t_batch)

  for key in ('W1', 'b1', 'W2', 'b2') :
    net.params[key] -= learning_late * grad[key]

  loss = net.loss(x_batch, t_batch)
  train_loss_list.append(loss)

  if i % iter_per_epoch == 0 :
    train_acc = net.accuracy(x_batch, t_batch)
    train_acc_list.append(train_acc)

    test_acc = net.accuracy(x_batch, t_batch)
    test_acc_list.append(test_acc)

    print('train_acc: ', train_acc, '. test_acc: ', test_acc)

plt.plot(train_loss_list)
plt.show()
