# input data, PCA, cross_validation, classifier_compare
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# input : {file_name} as python string with the format consists of / and . example: "/acb/123.txt", "123.txt"
# output: result as python string example: "123"
def trimFileName(file_name):
	return file_name[file_name.rfind("/")+1:file_name.find(".")]

# input : None
# output: None
def main():
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
	print ("Test size Percentage: %.3f" % percentage)

	# Read from file
	print ("Reading the file...")
	input_res = read_csv(os.path.expanduser(file_name), nrows=3000)  # load pandas dataframe
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
	
	# initialize testing x and y's
	sss = StratifiedShuffleSplit(y, 3, test_size=percentage, random_state=0)
	for train_index, test_index in sss:
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]

		# PCA
		pca = PCA(n_components=200)
		shape = X_train.shape
		X_train = pca.fit_transform(X_train)
		X_test = pca.transform(X_test)
		shape = X_test.shape
		temp = pca.explained_variance_ratio_
		print(temp)
		print(np.sum(temp))
	
		print ("Start building classifiers. This could take a while.")
		names = ["GauNB", "SVClinear", "LDA"]
		length = len(names)
		classifiers = [GaussianNB(), svm.SVC(kernel='linear', cache_size = 3000, probability=True, random_state=42), LDA()]
		assert (length == len(classifiers)), "legnth of classifiers(%d) did not match names(%d)"% (len(classifiers), length)
		# Learn to predict each class against the other
		for i in range(length):
			start = time.clock()
			y_pred = classifiers[i].fit(X_train, y_train).predict(X_test)
			elapsed_time = time.clock() - start
			print("Time to build classifier for %s: %.2f secs." % (names[i], elapsed_time))
			# Compute performance measurements
			print("Accuracy Score: %.3f" % metrics.accuracy_score(y_test, y_pred))
			print("Classification report for classifier %s:\n%s\n" % (classifiers[i], metrics.classification_report(y_test, y_pred)))
			print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))


main()

