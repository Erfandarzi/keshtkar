from keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np


from google.colab import files

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import np_utils
import pandas as pd


#setting parameters
batch_size = 32
nb_classes = 10
nb_epoch = 30
lr=0.001


# Load MNIST dataset
(X_in, y_train), (X_test, y_test) = mnist.load_data()

pca = PCA(n_components=32)
X_in=pca.fit_transform(X_in.reshape(60000,784))
X_test=pca.fit_transform(X_test.reshape(10000,784))


#Splitting Data
X_val = X_in[40000:].reshape(20000, 32)
X_train = X_in[:40000].reshape(40000, 32)
X_test = X_test.reshape(10000, 32)


X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
X_val /= 255


#Splitting the labels
Y_val = np_utils.to_categorical(y_train, nb_classes)[40000:]
Y_train = np_utils.to_categorical(y_train, nb_classes)[:40000]
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

model.compile(optimizer=RMSprop(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
history = model.fit(X_train, Y_train,validation_data=(X_val,Y_val),nb_epoch=nb_epoch, batch_size=batch_size, verbose=1)

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# or save to csv: 
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
files.download(' lr='+str(lr)+'batch='+str(batch_size)+'.csv') 
  
# Evaluate
evaluation = model.evaluate(X_test, Y_test, verbose=1)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))
