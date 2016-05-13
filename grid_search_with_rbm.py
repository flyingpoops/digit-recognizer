import sys, os, math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import sklearn.cross_validation as cval

from pandas.io.parsers import read_csv


###############################################################################
# Setting up

# Make a new folder to save results
newDir = str(os.getcwd())+"/analysis"
if not os.path.exists(newDir):
	os.makedirs(newDir)

# Transform user input
file_name = str(sys.argv[1]) #iris1.csv
print ("Name of file: %s" % file_name)
cut_pt = int(sys.argv[2]) #the index of the first x 1102
print ("Cut point: %d (preprocessed feature vector)" % cut_pt)
percentage = float(sys.argv[3]) #testing data percentage 0.25
print ("Percentage: %.3f" % percentage)

# Read from file
print ("Reading the file...")
input_res = read_csv(os.path.expanduser(file_name), nrows=10000)  # load pandas dataframe
#input_res = np.genfromtxt(file_name, delimiter=',', skiprows=1)
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

##############################################################################
# PCA

# Initialize parameters for PCA
degree = 100 # number of dimessions left

# Run PCA
pca = PCA(n_components=degree)
x = pca.fit_transform(x)
temp = pca.explained_variance_ratio_

# Print performance result
print("Explained variance ratio of each vector")
print(temp)
print("Cumulative explained variance ratio: %0.4f" % np.sum(temp))

##############################################################################
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = cval.train_test_split(x, y, test_size=0.5, random_state=0)

##############################################################################
# Train classifiers

# Intialize Classifier
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
rbm.n_components = 100

# Initialize grid_search range and other parameters
learning_rate_range = np.array([0.01,0.1,0.2,0.3])
n_iter_range = np.array([10,20,30])
c_range = np.array([1000,5000,10000])
param_grid = dict(rbm__learning_rate=learning_rate_range, rbm__n_iter=n_iter_range, logistic__C=c_range)
cv = cval.StratifiedShuffleSplit(y_train, n_iter=2, test_size=0.25, random_state=42)

# Run grid search
start = time.clock()
grid = GridSearchCV(classifier, param_grid=param_grid, cv=cv, n_jobs=-1, error_score=0) # change classifier here
grid.fit(X_train, y_train)
elapsed_time = time.clock() - start

# Print performance result
print("Time to build classifier: %.2f secs." % (elapsed_time))
print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
print(grid.best_estimator_)

##############################################################################
# Test classifiers
y_true, y_pred = y_test, grid.predict(X_test)

# Classification report
print("Accuracy Score: %.3f" % metrics.accuracy_score(y_test, y_pred))
print("Detailed classification report:")
print(classification_report(y_true, y_pred))

##############################################################################
# Save classifiers

from sklearn.externals import joblib
joblib.dump(grid.best_estimator_, newDir+'/abc.pkl', compress = 1)
