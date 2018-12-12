from __future__ import print_function
import cv2
import numpy as np

import os

import scipy.ndimage

from skimage.feature import hog

from skimage import data, color, exposure

from sklearn.model_selection import  train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
from characters import characters
from sklearn.decomposition import PCA
#%matplotlib inline

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
   # plot(X,Y,1)
    

    #print('\nTotal Training Set size: ', m_train)
    #print('Total Test Set size: ', m_test)
    return x_train, x_test, y_train, y_test

x_train_full, x_test_full, y_train_full, y_test_full = main()
img_rows, img_cols = 32,32
#print(y_test)

x_train = x_test_full[0:10000]
y_train = y_test_full[0:10000] 
x_test = x_test_full[10000:12000]
y_test = y_test_full[10000:12000] 
print(x_test.shape)
print(y_test.shape)

y_train = y_train.ravel()
y_test = y_test.ravel()

pca = PCA(n_components=150)

pca.fit(x_train)
var=pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)
print(var1)
plt.plot(var1)
plt.show()
