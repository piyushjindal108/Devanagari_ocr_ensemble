
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
from sklearn.decomposition import PCA
from scipy import spatial
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import RandomizedPCA
from sklearn.pipeline import make_pipeline
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

x_train = x_train_full[0:72500]
y_train = y_train_full[0:72500]
x_test = x_test_full[0:18000]
y_test = y_test_full[0:18000]
print(x_test.shape)
print(x_train.shape)

y_train = y_train.ravel()
y_test = y_test.ravel()

'''

pca = PCA(n_components=200, svd_solver='randomized')
x_train=pca.fit_transform(x_train)
pca1 = PCA(n_components=200, svd_solver='randomized')
x_test=pca1.fit_transform(x_test)
'''


df_1 = []

for i in range(0,x_train.shape[0]) :

    df1= hog(np.reshape(x_train[i],(32,32)), orientations=8, pixels_per_cell=(4,4), cells_per_block=(8, 8))
    df_1.append(df1)
x_train  = np.array(df_1, 'float64')

df_2 = []


for i in range(0,x_test.shape[0]) :

    df2= hog(np.reshape(x_test[i],(32,32)), orientations=8, pixels_per_cell=(4,4), cells_per_block=(8, 8))
    df_2.append(df2)
x_test  = np.array(df_2, 'float64')
print('Shape of Training data after HOG is: ' + str(x_train.shape))
print('Shape of Testing data after HOG is: ' + str(x_test.shape))

model = SVC(kernel='linear', class_weight='balanced',cache_size=3000,C=100)
#model = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='auto'), n_jobs=-1)
#model = make_pipeline(svc)
#param_grid = {'svc__C': [1],
#              'svc__gamma': [0.001]}
#grid = GridSearchCV(model, param_grid)

#grid.fit(x_train, y_train)
model.fit(x_train, y_train)              
#print(grid.best_params_)
#model = grid.best_estimator_
yfit = model.predict(x_test)
filename = 'finalized_model_svm.sav'
#joblib.dump(model, open(filename, 'wb'))
'''
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(x_test[i+32].reshape(32, 32), cmap='bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(yfit[i+32]-1,
                   color='black' if yfit[i+32] == y_test[i+32] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14);
print ('The score is ' + str(sum(yfit==y_test)/y_test.size))
predicted = []
actual = []
right = []
fig.show()
'''
indices = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45])

'''
for i in range (18000):
    if(yfit[i] == y_test[i])
    right.append(yfit[i]-1)
    else
    predicted.append(yfit[i]-1)
    actual.append(y_test[i]-1)
predicteda = np.asarray(predicted)
actuala = np.asarray(actual)
righta = np.asarray(right)
'''
print ('The score is ' + str(sum(yfit==y_test)/y_test.size))
np.set_printoptions(threshold=10000)
print("Classification report for classifier %s:\n%s\n"
      % (model, metrics.classification_report(y_test, yfit)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, yfit))
'''
def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap != None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)

cm_analysis(y_test, yfit, filename = 'fig', labels=indices+1, ymap=None, figsize=(10,10))
'''
