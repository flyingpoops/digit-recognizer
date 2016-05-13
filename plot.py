import sys, os, math
import time
import numpy as np
from pandas.io.parsers import read_csv

from sklearn.decomposition import PCA

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn import metrics
import sklearn.svm as svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.lda import LDA
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

cut_pt = 1
print ("Reading the file...")
input_res = read_csv(os.path.expanduser("input/train.csv"), nrows=3000)  # load pandas dataframe
input_res = input_res.as_matrix()
shape = input_res.shape
number_of_rows = shape[0]
number_of_columns = shape[1]
number_of_fv = number_of_columns - cut_pt
print ("Number of rows: %d (document)" % number_of_rows)
print ("Number of columns: %d (feature vector(preprocessed) + topics class labels(preprocessed))" % number_of_columns)
print ("Number of class_labels: %d" % number_of_fv)

# initialize training x and y's
x = input_res[:,cut_pt:number_of_columns]
y = input_res[:,0:cut_pt].transpose().ravel()

x = x / 255.
data = x[0]
print (data)
print (data.shape[0])
img = data.reshape(28, 28)
img = img.astype(np.float32)

plt.imshow(img, cmap="gray")
plt.show()
