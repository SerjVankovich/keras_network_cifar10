from keras.datasets import cifar10
import keras
from keras.utils import np_utils


path = 'cat2.jpeg'


TRAINED_PATH = 'trained_network.hdf5'

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

model = keras.models.load_model(TRAINED_PATH)


model.fit(X_train, Y_train, batch_size=32, epochs=25, validation_split=0.1, shuffle=True)
scores = model.evaluate(X_test, Y_test)
model.save(TRAINED_PATH)

print('Точность:', scores[1] * 100, "%")