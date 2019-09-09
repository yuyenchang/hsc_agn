#########################
##  agn_keras          ##
##  Yu-Yen Chang       ##
##  2019.08.25         ##
#########################

import numpy as np
import os.path
from astropy.io import fits, ascii
from astropy.table import Table, Column, MaskedColumn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.pyplot import matshow

####################################################################################################

class AGN:

	kind = 'AGN'

	def __init__(self, name):
		self.name = name    # instance variable unique to each instance

	def ml_keras(self, agnf0, mag00, inp):

		ep = 100            # epochs for model fit
		## ========== Keras model ==========

		## split data into X and y
		X = np.zeros((len(mag00), inp, 1))
		X[:,:,0] = mag00
		Y = agnf0
		
		## split data into train and test sets
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.33, random_state=4)

		## imbalance class (Over-sampling )
		i_agn = np.where(self.y_train == 1) 
		i_ran = np.random.choice(i_agn[0], self.y_train.size)
		self.y_train = np.append(self.y_train, self.y_train[i_ran])
		self.X_train = np.concatenate((self.X_train, self.X_train[i_ran,:]))

		## preprocess input data
		self.X_train = self.X_train.astype('float32')
		self.X_test = self.X_test.astype('float32')
		xmax=np.nanmax(self.X_train)
		xmin=np.nanmin(self.X_train)
		self.X_train = (self.X_train-xmin)/(xmax-xmin)
		self.X_test = (self.X_test-xmin)/(xmax-xmin)
		 
		## preprocess class labels: AGN fraction
		self.Y_train = np_utils.to_categorical(self.y_train, 2)
		self.Y_test = np_utils.to_categorical(self.y_test, 2)

		## define model architecture
		model = Sequential()
		model.add(Convolution1D(32, 3, border_mode='same', input_shape=(inp, 1)))
		model.add(Convolution1D(32, 3, border_mode='same'))
		model.add(Flatten())
		model.add(Dense(128, activation='relu'))
		model.add(Dropout(0.5))
		model.add(Dense(2, activation='softmax'))

		## compile model
		model.compile(loss='categorical_crossentropy',
		              optimizer='RMSprop',
		              metrics=['accuracy'])
		             
		## fit model on training data
		history = model.fit(self.X_train, self.Y_train, validation_split=0.33, epochs=ep, batch_size=32, verbose=0)

		## evaluate model on test data
		score = model.evaluate(self.X_test, self.Y_test, verbose=0)

		## predict Y value vs real Y test value
		self.Y_predict = model.predict(self.X_test, batch_size=32, verbose=0, steps=None)
		self.y_predict=np.argmax(self.Y_predict ,axis=1)

		## ROC curve
		self.fpr = dict()
		self.tpr = dict()
		self.roc_auc = dict()
		for i in range(2):
		    self.fpr[i], self.tpr[i], _ = roc_curve(self.y_test, self.Y_predict[:,1])
		    self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])

		## Precision Recall curve
		self.average_precision = average_precision_score(self.y_test,  self.Y_predict[:,1])
		self.precision, self.recall, _ = precision_recall_curve(self.y_test, self.Y_predict[:,1])
	    
	    ## Ouputs
		self.acc = '{:02.5f}'.format(metrics.accuracy_score(self.y_test, self.y_predict))
		self.p = '{:02.5f}'.format(metrics.precision_score(self.y_test, self.y_predict, average='macro'))
		self.r = '{:02.5f}'.format(metrics.recall_score(self.y_test, self.y_predict, average='macro'))
		self.f1 = '{:02.5f}'.format(metrics.f1_score(self.y_test, self.y_predict, average='macro'))
		self.roc = '{:02.5f}'.format(self.roc_auc[1])
		self.ap = '{:02.5f}'.format(self.average_precision)

		return [self.acc, self.p, self.r, self.f1, self.roc, self.ap]

####################################################################################################

## read fits file & data
hdul = fits.open('_agn_sel.fits')
data = hdul[1].data
header = hdul[1].header
print(ascii.read("_agn_sel.cat") )

ID = data['NUMBER']
gflux = data['g_cmodel_flux']
rflux = data['r_cmodel_flux']
iflux = data['i_cmodel_flux']
zflux = data['z_cmodel_flux']
yflux = data['y_cmodel_flux']
gflux_e = data['g_cmodel_fluxsigma']
rflux_e = data['r_cmodel_fluxsigma']
iflux_e = data['i_cmodel_fluxsigma']
zflux_e = data['z_cmodel_fluxsigma']
yflux_e = data['y_cmodel_fluxsigma']
redshift = data['photoz_best']
w1mag = data['w1mpro']
w2mag = data['w2mpro']
w1mag_e = data['w1sigmpro']
w2mag_e = data['w2sigmpro']

## choose data sets
##------HSC only------
#mag00 = np.column_stack((gflux, rflux, iflux, zflux, yflux, gflux_e, rflux_e, iflux_e, zflux_e, yflux_e, redshift)) 
#inp = 11
##------HSC+W12------
mag00 = np.column_stack((gflux, rflux, iflux, zflux, yflux, gflux_e, rflux_e, iflux_e, zflux_e, yflux_e, redshift, 
                         w1mag, w2mag, w1mag_e, w2mag_e))
inp = 15
mag00.shape

#---XAGN---
indtf = np.where(data['agn_x'] == 1)
agnf0 = np.zeros(len(redshift))
agnf0[indtf] = 1
agn1 = AGN('XAGN')
output_agn1 = agn1.ml_keras(agnf0, mag00, inp)

#---IRAGN---
indtf = np.where(data['agn_ir'] == 1)
agnf0 = np.zeros(len(redshift))
agnf0[indtf] = 2
agn2 = AGN('IRAGN')
output_agn2 = agn2.ml_keras(agnf0, mag00, inp)

#---RAGN---
indtf = np.where(data['agn_r'] == 1)
agnf0 = np.zeros(len(redshift))
agnf0[indtf] = 3
agn3 = AGN('RAGN')
output_agn3 = agn3.ml_keras(agnf0, mag00, inp)

output = Table(rows=[output_agn1, output_agn2, output_agn3], 
               names=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUROC', 'AP'])
print(output)

plt.plot(agn1.fpr[1], agn1.tpr[1], color='darkorange', label='ROC curve (area = %0.2f)' % agn1.roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

plt.step(agn1.recall, agn1.precision, color='b', alpha=0.2, where='post')
plt.fill_between(agn1.recall, agn1.precision, alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(agn1.average_precision))
plt.show()

mt = metrics.confusion_matrix(agn1.y_test, agn1.y_predict)
m = mt/np.max(mt)
plt.matshow(m,cmap=plt.get_cmap('rainbow'), norm=colors.LogNorm(vmin=1e-4, vmax=1))
plt.colorbar()
plt.title('Normalized confusion matrix', pad=10)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'])
plt.show()
