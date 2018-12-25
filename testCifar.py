from keras.datasets import cifar10
import numpy as np
from PIL import Image

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
img = Image.fromarray(np.uint8(X_train[0]))
imgrgb = img.convert('RGB')
imgrgb.show()

print(imgrgb)

