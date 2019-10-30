from google.colab import files

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#setting parameters
batch_size = 32
nb_epoch = 300
lr=0.001


# Load MNIST dataset
(train_data,train_targets),(test_data,test_targets)=boston_housing.load_data(test_split=0.8)

#Splitting Data
# train_data= train_data.astype('float32')
# test_data =test_data.astype('float32')

# test_targets =test_targets.astype('float32')
# test_targets=test_targets.astype('float32')

print(test_targets)

# Deep Multilayer Perceptron model
model = Sequential()
model.add(Dense(output_dim=128, input_dim=13, init='normal'))
model.add(Activation('relu'))
# model.add(Dense(output_dim=64, input_dim=64, init='normal'))
# model.add(Activation('relu'))
# model.add(Dense(output_dim=64, input_dim=64, init='normal'))
# model.add(Activation('relu'))
model.add(Dense(output_dim=1, input_dim=128, init='normal'))
model.add(Activation('relu'))

model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(train_data, train_targets,validation_data=(test_data,test_targets),nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)
print('MSE=',np.sqrt(mean_squared_error(test_targets, model.predict(test_data)[:, 0])))
print('Maximum error=',np.amax(np.abs(test_targets, model.predict(test_data)[:, 0])))# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
