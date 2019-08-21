#########################
##  hsc_agn_xgboost.py ##
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
from xgboost import DMatrix
import xgboost as xgb

####################################################################################################

## read fits file & data
hdul = fits.open('agn_all_s17a_wide_cos_sel_z10.fits')
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
fn=['g','r','i','z','y','g_e','r_e','i_e','z_e','y_e','redshift']
##------HSC+W12------
mag00=np.column_stack((gflux,rflux,iflux,zflux,yflux,gflux_e,rflux_e,iflux_e,zflux_e,yflux_e,redshift,w1mag,w2mag,w1mag_e,w2mag_e))
fn=['g','r','i','z','y','g_e','r_e','i_e','z_e','y_e','redshift','w1','w2','w1_e','w2_e']
##------HSC+WISE------
#mag00=np.column_stack((gflux,rflux,iflux,zflux,yflux,gflux_e,rflux_e,iflux_e,zflux_e,yflux_e,redshift,w1mag,w2mag,w3mag,w4mag,w1mag_e,w2mag_e,w3mag_e,w4mag_e))
#fn=['g','r','i','z','y','g_e','r_e','i_e','z_e','y_e','redshift','w1','w2','w3','w4','w1_e','w2_e','w3_e','w4_e']
##------WISE only------
#mag00=np.column_stack((w1mag,w2mag,w3mag,w4mag,w1mag_e,w2mag_e,w3mag_e,w4mag_e))
#fn=['w1','w2','w3','w4','w1_e','w2_e','w3_e','w4_e']

test_size = 0.33  # for test set
seed = 4          # for test set
num = 100         # num_round for training iterations
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

	## ========== XGBoost model ==========

	## split data into X and Y
	X = mag00
	Y = agnf0

	## split data into train and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

	## add all AGN
	i_agn = np.where(agnf0==1)
	y_train = np.append(y_train,agnf0[i_agn])
	X_train = np.concatenate((X_train,mag00[i_agn[0],:]))

	## define parameters
	dtrain = DMatrix(X_train, label=y_train,feature_names=fn)
	dtest = DMatrix(X_test, label=y_test, feature_names=fn)
	param = {
	    'max_depth': 3,                 # the maximum depth of each tree
	    'eta': 0.3,                     # the training step for each iteration
	    'silent': 1,                    # logging mode - quiet
	    'objective': 'multi:softprob',  # error evaluation for multiclass training
	    'num_class': 2}                 # the number of classes that exist in this datset
	num_round = num                     # the number of training iterations
	bst = xgb.train(param, dtrain, num_round)

	## predict Y value vs real Y test value
	Y_predict = bst.predict(dtest)
	y_predict = np.argmax(Y_predict ,axis=1)

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

