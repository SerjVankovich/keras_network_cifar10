import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

np.random.seed(42)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print(y_train )
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(32, 32, 3), activation='relu',
                 data_format='channels_last'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(32, 3, 3), activation='relu',
                 data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', data_format='channels_last'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='valid', activation='relu', data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=32, epochs=1, validation_split=0.1, shuffle=True)

scores = model.evaluate(X_test, y_test, verbose=0)


print("Точность:", scores[1] * 100)
