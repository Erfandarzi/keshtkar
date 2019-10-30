from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
import numpy as np

# Hyper parameters
batch_size = 128
nb_epoch = 30

# Parameters for MNIST dataset
img_rows, img_cols = 28, 28

#Loading dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


#Autoencoder architecture
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)  # Multiple encoding
decoded = Dense(64, activation='relu')(encoded)  # and decoding layers.
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

#Autoencoder compile
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()


encoder = Model(autoencoder.input, autoencoder.layers[-4].output)
encoder.summary()

#Autoencoder Fitting

autoencoder.fit(x_train, x_train,
                nb_epoch=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1,
                validation_data=(x_test, x_test))
            
                
encoderdecoder=autoencoder.predict(x_train)
encoded=encoder.predict(x_train)


#Visualizing the decoded and encoded image
import cv2
import matplotlib.pyplot as plt
plt.imshow(np.reshape(x_train[300],(28,28)))
plt.show()

plt.imshow(np.reshape(encoderdecoder[300],(28,28)))
plt.show()
print((encoded))


#Building the mlp but with decoded input this time

from google.colab import files

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd

batch_size = 4
nb_classes = 10
nb_epoch = 30

# Load MNIST dataset for test
(__, y_train), (X_test, y_test) = mnist.load_data()


X_train= encoded

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Deep Multilayer Perceptron model
model = Sequential()
model.add(Dense(output_dim=512, input_dim=32, init='normal'))
model.add(Activation('relu'))
model.add(Dense(output_dim=512, input_dim=512, init='normal'))
model.add(Activation('relu'))
model.add(Dense(output_dim=512, input_dim=512, init='normal'))
model.add(Activation('relu'))
model.add(Dense(output_dim=10, input_dim=512, init='normal'))
model.add(Activation('softmax'))

model.compile(optimizer=RMSprop(lr=0.001, rho=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
