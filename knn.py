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
from scipy import spatial
import _pickle as pickle
from sklearn.externals import joblib

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

x_train = x_train_full[0:73500]
y_train = y_train_full[0:73500]
x_test = x_test_full[0:18000]
y_test = y_test_full[0:18000]
print(x_test.shape)
print(y_test.shape)

y_train = y_train.ravel()
y_test = y_test.ravel()

df_1 = []

for i in range(0,x_train.shape[0]) :

    df1= hog(np.reshape(x_train[i],(32,32)), orientations=8, pixels_per_cell=(8,8), cells_per_block=(4, 4))
    df_1.append(df1)
x_train  = np.array(df_1, 'float64')

df_2 = []


for i in range(0,x_test.shape[0]) :

    df2= hog(np.reshape(x_test[i],(32,32)), orientations=8, pixels_per_cell=(8,8), cells_per_block=(4, 4))
    df_2.append(df2)
x_test  = np.array(df_2, 'float64')
print('Shape of Training data after HOG is: ' + str(x_train.shape))
print('Shape of Testing data after HOG is: ' + str(x_test.shape))

'''

#-----------------------------------------------------------------------Failed PCA-----------------------------------------------------------------------------------------------------------

pca = PCA(n_components=400)
pca.fit(x_train)
x_train=pca.fit_transform(x_train)
pca = PCA(n_components=400)
pca.fit(x_test)
x_test=pca.fit_transform(x_test)
var= pca.explained_variance_ratio_
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print (var1)
plt.plot(var1)
plt.show()
print('Train shape after pca is ' + str(x_train.shape))
print('Test shape after pca is ' + str(x_test.shape))

#print (x_test)
'''
#-------------------------------------------------------------------------knn--------------------------------------------------------------------------------------------------------------
#print (model_score)
#print(y_test)


knn = KNeighborsClassifier(n_neighbors=5,n_jobs = -1,metric = 'braycurtis')
knn.fit(x_train, y_train)
model_score = knn.score(x_test, y_test)
#joblib.dump(knn, 'knn_model.pkl')
print('model score for K = 5' +  ' is ' + str(model_score))
filename = 'finalized_model.sav'
#joblib.dump(knn, open(filename, 'wb'))


            


