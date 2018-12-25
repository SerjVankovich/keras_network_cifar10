import sys

import numpy as np
import keras
from PIL import Image

path = 'truck9.png'

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


TRAINED_PATH = 'trained_network.hdf5'

img = Image.open(path)
img.thumbnail((32, 32), Image.ANTIALIAS)

data = np.array(img).astype('float32') / 255
data = np.array([data])

model = keras.models.load_model(TRAINED_PATH)
x = model.predict(x=data)
index = np.argmax(x[0])
print(classes[index])
print(round(x[0][index] * 100, 2))
