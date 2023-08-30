import sys, os
sys.path.append(os.path.join(sys.path[0], '../../'))

import numpy as np
from PIL import Image
from dataset.mnist import load_mnist

def img_show (img) :
  pil_img = Image.fromarray(np.uint8(img))
  pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(
    flatten = True,
    normalize = False,
    one_hot_label = False
  )

img = x_train[0]
label = t_train[0]

# print label
print(label)

# reshape
print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

# show img
img_show(img)
