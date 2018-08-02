from __future__ import print_function
import cv2
import numpy as np

import os

import scipy.ndimage

from skimage.feature import hog

from skimage import data, color, exposure

from sklearn.model_selection import  train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from characters import characters
from scipy import spatial
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import _pickle as pickle
from sklearn.externals import joblib
from keras.models import Sequential
from keras.optimizers import SGD
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from characters import characters
from keras.models import model_from_json

#from sklearn.grid_search import GridSearchCV
verbose = True

np.random.seed(1)
def mydist(x, y):
    return spatial.distance.cosine(x, y)
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
   # X = scale(X)
    x_train, x_test, y_train, y_test = split(X, Y)

    #m_train = y_train.shape[0]
   # m_test =y_test.shape[0]
   # plot(X,Y,1)
    

    #print('\nTotal Training Set size: ', m_train)
    #print('Total Test Set size: ', m_test)
    return x_train, x_test, y_train, y_test

x_train_full, x_test_full, y_train_full, y_test_full = main()
img_rows, img_cols = 32,32
#print(y_test)


x_test = x_test_full[0:18000]
y_test = y_test_full[0:18000]
print(x_test.shape)


y_test = y_test.ravel()

#---------------------------------------------------------------------------------hog for svm-----------------------------------------------------------------------------------------------------



df_2 = []


for i in range(0,x_test.shape[0]) :

    df2= hog(np.reshape(x_test[i],(32,32)), orientations=8, pixels_per_cell=(4,4), cells_per_block=(8, 8))
    df_2.append(df2)
x_test_hogged_svm  = np.array(df_2, 'float64')
print('Shape of Testing data after HOG-svm is: ' + str(x_test_hogged_svm.shape))
#---------------------------------------------------------------------------------hog for knn-----------------------------------------------------------------------------------------------------



df_3 = []


for i in range(0,x_test.shape[0]) :

    df3= hog(np.reshape(x_test[i],(32,32)), orientations=8, pixels_per_cell=(8,8), cells_per_block=(4, 4))
    df_3.append(df3)
x_test_hogged_knn  = np.array(df_3, 'float64')
print('Shape of Testing data after HOG-knn is: ' + str(x_test_hogged_knn.shape))
#-------------------------------------------------------------------------------------SVC------------------------------------------------------------------------------------------------------
filename_svm = 'finalized_model_svm.sav'
loaded_model_svm = joblib.load(filename_svm)
y_svc = loaded_model_svm.predict(x_test_hogged_svm[0:18000])
print('SVC done')
print ('The score of SVC is ' + str(sum(y_svc==y_test)/y_test.size))
#-----------------------------------------------------------------------------------knn----------------------------------------------------------------------------------------------------
filename_knn = 'finalized_model_knn.sav'
loaded_model_knn = joblib.load(filename_knn)
y_knn = loaded_model_knn.predict(x_test_hogged_knn[0:18000])
print('knn done')
model_score = loaded_model_knn.score(x_test_hogged_knn, y_test)
print('The score for K = 5(KNN)' +  ' is ' + str(model_score))
#------------------------------------------------------------------------------------cnn-----------------------------------------------------------------------------------------------------

x_test_cnn = x_test_full[0:18000]
y_test_cnn = y_test_full[0:18000] - 1
if K.image_data_format() == 'channels_first':
   
    x_test_cnn = x_test_cnn.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    
    x_test_cnn = x_test_cnn.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_test_cnn = x_test_cnn.astype('float32')
y_test_cnn = keras.utils.to_categorical(y_test_cnn, 46)
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model.h5")
print("Loaded model from disk")
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
loaded_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.sgd(),
              metrics=['accuracy'])

score = loaded_model.evaluate(x_test_cnn, y_test_cnn, verbose=0)
print('Test loss for CNN', score[0])
print('The Score for CNN is ', score[1])
y_cnn = loaded_model.predict_classes(x_test_cnn[0:18000])
y_cnn = y_cnn + 1
#------------------------------------------------------------------------------------ensemble-----------------------------------------------------------------------------------------------
ty = np.zeros((3,18000))
ty[0] = y_svc
ty[1] = y_knn
ty[2] = y_cnn
print('The correlation b/w svc and knn is' + str(np.corrcoef(y_svc,y_knn)))
print('The correlation b/w svc and cnn is' + str(np.corrcoef(y_svc,y_cnn)))
print('The correlation b/w knn and cnn is' + str(np.corrcoef(y_knn,y_cnn)))



