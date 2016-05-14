import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,lib.cnmem=1,dnn.enabled=False"
import pandas as pd
import time

##########################################################
# Input varialbles
flag = 1 #0 for sklearn and 1 for keras
classifier_file_name = 'model/cnn2.json' #'C:\kaggle\dr\rfab.pkl' for sklearn
weight_file_name = 'model/weights.020-0.992.hdf5'
output_file_name = '789.csv'

if flag == 0:
	from sklearn.externals import joblib
	##########################################################
	# Reading Data
	print ("Reading Test Data")
	test = pd.read_csv('input/test.csv')
	##########################################################
	# Load Classifier
	print ("Loading Classifier")
	clf = joblib.load(classifier_file_name)
	##########################################################
	# Test classifier
	print ("Generating results to submit")
	y = clf.predict(test)

else:
	from keras.models import model_from_json
	##########################################################
	# Reading Data
	print ("Reading Test Data")
	test = pd.read_csv('input/test.csv').values
	testX = test.reshape(test.shape[0], 1, 28, 28)
	testX = testX.astype('float32')
	testX /= 255.0
	##########################################################
	# Load Classifier
	print ("Loading Classifier")
	fo = open(classifier_file_name, "r")
	model = model_from_json(fo.read()) 
	fo.close()
	model.load_weights(weight_file_name)
	model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])
	##########################################################
	# Test classifier
	print ("Generating results to submit")
	y = model.predict_classes(testX)

##########################################################
print ("Writing to file")
predictions = pd.DataFrame(data=y,columns=["label"])
predictions["ImageId"] = list(range(1,len(test)+1))

predictions.to_csv(output_file_name,index=False)