#########################
##  hsc_agn_keras.py   ##
##  Yu-Yen Chang       ##
##  2019.08.01         ##
#########################

import numpy as np
import os.path
from astropy.io import fits
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

####################################################################################################

## read fits file & data
hdul = fits.open('agn_all_s17a_wide_cos_sel.fits')
data = hdul[1].data
header=hdul[1].header
ID = data['NUMBER']
ra=data['alpha_j2000']
dec=data['delta_j2000']
gflux=data['g_kronflux_flux']
rflux=data['r_kronflux_flux']
iflux=data['i_kronflux_flux']
zflux=data['z_kronflux_flux']
yflux=data['y_kronflux_flux']
gflux_e=data['g_kronflux_fluxsigma']
rflux_e=data['r_kronflux_fluxsigma']
iflux_e=data['i_kronflux_fluxsigma']
zflux_e=data['z_kronflux_fluxsigma']
yflux_e=data['y_kronflux_fluxsigma']
redshift=data['photoz_best']
w1mag=data['w1mpro']
w2mag=data['w2mpro']
w3mag=data['w3mpro']
w4mag=data['w4mpro']
w1mag_e=data['w1sigmpro']
w2mag_e=data['w2sigmpro']
w3mag_e=data['w3sigmpro']
w4mag_e=data['w4sigmpro']

## choose data sets
##------HSC only------
mag00=np.column_stack((gflux,rflux,iflux,zflux,yflux,gflux_e,rflux_e,iflux_e,zflux_e,yflux_e,redshift)) 
inp=11
##------HSC+W12------
mag00=np.column_stack((gflux,rflux,iflux,zflux,yflux,gflux_e,rflux_e,iflux_e,zflux_e,yflux_e,redshift,w1mag,w2mag,w1mag_e,w2mag_e))
inp=15
##------HSC only------
#mag00=np.column_stack((gflux,rflux,iflux,zflux,yflux,gflux_e,rflux_e,iflux_e,zflux_e,yflux_e,redshift,w1mag,w2mag,w3mag,w4mag,w1mag_e,w2mag_e,w3mag_e,w4mag_e))
#inp=19
##------HSC+W12------
#mag00=np.column_stack((w1mag,w2mag,w3mag,w4mag,w1mag_e,w2mag_e,w3mag_e,w4mag_e))
#inp=8

test_size = 0.33  # for test set
seed = 4          # for test set
ep=100            # epochs for model fit
for ii in range(3):

	## AGN types
	if ii == 0:  #---XAGN---
		indtf=np.where(data['agn_cart'] == 1) 
		agnf0=np.zeros(len(redshift))
		agnf0[indtf]=1
	if ii == 1:  #---IRAGN---
		indtf=np.where((data['agn_cart'] == 1) & (data['redshift']<2.5) & (data['redshift']>0))
		agnf0=np.zeros(len(redshift))
		agnf0[indtf]=1
	if ii == 2:  #---RAGN---
		indtf=np.where(data['Radio_excess'] == 'true')
		agnf0=np.zeros(len(redshift))
		agnf0[indtf]=1

	inz=np.where((redshift >=0) )
	agnf=agnf0[inz[0]]
	mag0=mag00[inz[0],:]

	## ========== Keras model ==========

	## split data into X and y
	X = np.zeros((len(mag00),inp,1))
	X[:,:,0] = mag00
	Y = agnf
	
	## split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	## preprocess input data
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	xmax=np.nanmax(X_train)
	xmin=np.nanmin(X_train)
	X_train = (X_train-xmin)/(xmax-xmin)
	X_test = (X_test-xmin)/(xmax-xmin)
	 
	## preprocess class labels: AGN fraction
	Y_train = np_utils.to_categorical(y_train, 2)
	Y_test = np_utils.to_categorical(y_test, 2)

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
	history = model.fit(X_train, Y_train, validation_split=0.33, epochs=ep, batch_size=32, verbose=0)

	## evaluate model on test data
	score = model.evaluate(X_test, Y_test, verbose=0)

	## predict Y value vs real Y test value
	Y_predict = model.predict(X_test, batch_size=32, verbose=0, steps=None)
	y_predict=np.argmax(Y_predict ,axis=1)

	## ROC curve
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(2):
	    fpr[i], tpr[i], _ = roc_curve(y_test, Y_predict[:,1])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	## Precision Recall curve
	average_precision = average_precision_score(y_test,  Y_predict[:,1])
	precision, recall, _ = precision_recall_curve(y_test, Y_predict[:,1])
	ap=average_precision

	acc=metrics.accuracy_score(y_test, y_predict)
	p=metrics.precision_score(y_test, y_predict, average='macro')
	r=metrics.recall_score(y_test, y_predict, average='macro')
	f1=metrics.f1_score(y_test, y_predict, average='macro')
	roc=roc_auc[1]

	if ii == 0:  #---XAGN---
		print('XAGN &','%.4f' % acc,'&','%.4f' % p,'&','%.4f' % r,'&','%.4f' % f1,'&','%.4f' % roc,'&','%.4f' % ap,'\\\\')
	if ii == 1:  #---IRAGN---
		print('IRAGN &','%.4f' % acc,'&','%.4f' % p,'&','%.4f' % r,'&','%.4f' % f1,'&','%.4f' % roc,'&','%.4f' % ap,'\\\\')
	if ii == 2:  #---RAGN---
		print('RGN &','%.4f' % acc,'&','%.4f' % p,'&','%.4f' % r,'&','%.4f' % f1,'&','%.4f' % roc,'&','%.4f' % ap,'\\\\')
