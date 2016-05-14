# Simple Convolutional Nerual Network (keras part is mostly adapted from keras example at https://github.com/fchollet/keras/tree/master/examples)
import os
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32,cnmem=0.75"
# temp files will be possibly stored at C:\Users\David\AppData\Local\Theano as indicated in the terminal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import keras.models as models
import keras.utils.np_utils as kutils
import keras.callbacks as cb

from sklearn.cross_validation import train_test_split
import sklearn.metrics as metrics

import theano

##########################################################
# Input varialbles
nb_epoch = 12
batch_size = 128

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

theano.config.floatX = 'float32'

model_file_name = 'model/cnn.json'

##########################################################
# Additional Functions
def plotValidationCurves(param_range, train_accuracy, test_accuracy):
	train_mean = train_accuracy #train_mean = np.mean(train_accuracy, axis=1)
	#train_std = np.std(train_accuracy, axis=1)
	test_mean = test_accuracy #test_mean = np.mean(test_accuracy, axis=1)
	#test_std = np.std(test_accuracy, axis=1)
	plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
	#plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
	plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
	#plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
	plt.grid()
	#plt.xscale('log') #for log plot on x-axis
	plt.legend(loc='lower right')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.ylim([0.4, 1.0])
	plt.savefig('foo.png')

class ModelCheckpoint1(cb.Callback):
    
    def __init__(self, filepath, itera=1, monitor='val_loss', verbose=0, save_best_only=False, mode='auto'):

        self.itera = itera
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        filepath = self.filepath.format(epoch=epoch, **logs)
        if ((epoch+1)%self.itera == 0):
	        if self.save_best_only:
	            current = logs.get(self.monitor)
	            if current is None:
	                warnings.warn('Can save best model only with %s available, '
	                              'skipping.' % (self.monitor), RuntimeWarning)
	            else:
	                if self.monitor_op(current, self.best):
	                    if self.verbose > 0:
	                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
	                              ' saving model to %s'
	                              % (epoch, self.monitor, self.best,
	                                 current, filepath))
	                    self.best = current
	                    self.model.save_weights(filepath, overwrite=True)
	                else:
	                    if self.verbose > 0:
	                        print('Epoch %05d: %s did not improve' %
	                              (epoch, self.monitor))
	        else:
	            if self.verbose > 0:
	                print('Epoch %05d: saving model to %s' % (epoch, filepath))
	            self.model.save_weights(filepath, overwrite=True)

##########################################################
print ("Reading Data")
train = pd.read_csv('input/train.csv').values

trainX = train[:, 1:].reshape(train.shape[0], 1, img_rows, img_cols)
trainX = trainX.astype(theano.config.floatX)
trainX /= 255.0

trainY = train[:, 0]

trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.05, stratify=trainY)

train_Y = kutils.to_categorical(trainY)
test_Y = kutils.to_categorical(testY)
nb_classes = train_Y.shape[1]

##########################################################
print ("Building Neuro Network")
model = models.Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='valid', input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"])

with open(model_file_name, 'w') as fo: # save model
    fo.write(model.to_json())

##########################################################
print ("Training")
save_model = ModelCheckpoint1("model/weights.{epoch:03d}-{val_acc:.3f}.hdf5", itera=3, monitor='val_acc', verbose=0, save_best_only=False, mode='auto')
hist = model.fit(trainX, train_Y, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(testX, test_Y), verbose=1, callbacks=[save_model])

##########################################################
print ("Testing")

plotValidationCurves([w for w in range(nb_epoch)], hist.history['acc'], hist.history['val_acc'])

yPred = model.predict_classes(testX)

print ("Accuracy of the final model: %.3f" % (metrics.accuracy_score(testY, yPred)))
print ("Detailed Classification Report")
print (metrics.classification_report(testY, yPred))
print ("Confusion Matrix")
print (metrics.confusion_matrix(testY, yPred))

