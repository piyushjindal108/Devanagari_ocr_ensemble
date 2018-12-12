from __future__ import print_function
import cv2
import numpy as np
import _pickle as pickle
from keras.models import Sequential
from keras.optimizers import SGD
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from characters import characters
from keras.models import model_from_json

verbose = True

np.random.seed(1)

def load_data():
    if verbose:
        print('Loading dataset...')
    m = 92000
    data = np.load('dataset/dataset.npz')

    x_train = data['arr_0']
    y_train = data['arr_1'].reshape(78200, 1)
    x_test = data['arr_2']
    y_test = data['arr_3'].reshape(m - 78200, 1)

    X = np.vstack([x_train, x_test]).reshape(m, 1024)
    Y = np.vstack([y_train, y_test]).reshape(m, 1)

    return X, Y

def shuffle(X, Y):
    if verbose:
        print('Shuffling data...')
    from sklearn.utils import shuffle
    X, Y = shuffle(X, Y)
    return X, Y

def plot(X, Y, n = 0):
    print('The image is character: ', characters[Y[n][0] - 1])
    plt.imshow(X[n, :].reshape(32, 32), cmap = 'Greys')
    plt.show()

def scale(X, factor = 255):
    if verbose:
        print('Scaling Features...')
    return X*(1/255)

def split(x, y, ratio = 0.2):
    if verbose:
        print('Splitting dataset...')
    from sklearn.model_selection import train_test_split
    return train_test_split(x, y, test_size = ratio)

def main():
    X, Y = load_data()
    X, Y = shuffle(X, Y)
    X = scale(X)
    x_train, x_test, y_train, y_test = split(X, Y)

    m_train = y_train.shape[0]
    m_test =y_test.shape[0]
    plot(X,Y,1)
    

    print('\nTraining Set size: ', m_train)
    print('Test Set size: ', m_test)
    return x_train, x_test, y_train, y_test

x_train_full, x_test_full, y_train_full, y_test_full = main()

img_rows, img_cols = 32,32
#print(y_test)

x_train = x_test_full[0:10000]
y_train = y_test_full[0:10000] - 1
x_test = x_test_full[10000:12000]
y_test = y_test_full[10000:12000] - 1
print(x_test.shape)
print(y_test.shape)


batch_size = 128
num_classes = 46
epochs = 30

print(x_train.shape)    
    
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(np.min(y_train))

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
'''

model = Sequential()
    
model.add(Conv2D(64, kernel_size=(3, 3),
                     strides = (1,1), 
                     activation='relu',
                     padding='same',
                     kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     input_shape=input_shape,
                     name = 'conv01'));
model.add(Conv2D(64, kernel_size=(3, 3),
                     strides = (1,1), 
                     activation='relu',
                     padding='same',
                     kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     name = 'conv02'));
    
model.add(Conv2D(64, kernel_size=(3, 3),
                     strides = (1,1), 
                     activation='relu',
                     padding='same',
                     kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     name = 'conv03'));
    
model.add(MaxPooling2D((2, 2), strides = (2,2), name='max_pool'))
    
model.add(Conv2D(64, kernel_size=(3, 3),
                     strides = (1,1), 
                     activation='relu',
                     padding='same',
                     kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     name = 'conv11'));
    
model.add(Conv2D(64, kernel_size=(3, 3),
                     strides = (1,1), 
                     activation='relu',
                     padding='same',
                     kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     name = 'conv12'));
    
model.add(Conv2D(64, kernel_size=(3, 3),
                     strides = (1,1), 
                     activation='relu',
                     padding='same',
                     kernel_initializer= keras.initializers.RandomNormal(mean=0.0, stddev=0.1, seed=None),
                     name = 'conv13'));
    
model.add(MaxPooling2D((2, 2), strides = (2,2), name='max_pool1'))
    
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# check the loss function again
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
'''

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])

score = loaded_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_cnn = loaded_model.predict_classes(x_test)

#model.save_weights("model.h5")


