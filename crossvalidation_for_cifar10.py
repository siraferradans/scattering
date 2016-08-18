from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score, KFold, ShuffleSplit,cross_val_predict
from sklearn import metrics

from keras.datasets import cifar10
import numpy as np
def gethomogeneus_datast(X,y,d,n):
    num_per_class = np.int(n/d)
    ytrain=np.reshape(y,(y.shape[0],))

    X_out = []#np.zeros(n,X.shape[1],X.shape[2],X.shape[3])
    y_out = []
    for i_d in np.arange(d):
        indx = np.where(ytrain.ravel()==i_d)

        X_out.append(X[indx[0][0:num_per_class],:])
        y_out.append(ytrain[indx[0][0:num_per_class],])

    
    X_out = np.concatenate(X_out,axis=0)
    y_out = np.concatenate(y_out,axis=0)
    #print(X_out.shape)
    #print(y_out)
    return X_out,y_out


#(X_train_sm, y_train), (X_test_sm, y_test) = cifar10.load_data()
#X_train_sm.shape
n = 60000
from load_cifar10_db import load_images_cifar,load_scattering_cifar
X_train, y_train, X_test, y_test = load_scattering_cifar(num_images = 150000, J=3,L=8,m=1)
num_samples = X_train.shape[0]

Xtrain_1d = X_train.reshape((len(X_train),-1))
Xtest_1d = X_test.reshape((len(X_test),-1))
Xtest_1d.shape

J=3
L=8
print(32/2**(J-1))
print('num coefs:',(1+J*L)*3)
print(X_train.shape)

##### Gaussian kernel grid search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


gammavec = 10**np.arange(-0.5,0.5,0.1)# 
#gammavec = 10**np.arange(0.,0.15,0.05) #0.3-1.5(0.25) got 1.995 #(-1,0.5,0.1) got 0.5
gammavec.shape = (gammavec.shape[0])
Cvec = np.arange(0.6,1.5,0.20) #got 3.25 #np.arange(0.5,4,0.25) got 3.25
#Cvec = np.arange(4.1,4.4,0.1) #3-5(0.25))got 4.25  #4-5, 4.0 (0.5)
Cvec.shape=(Cvec.shape[0])
parameters = {"C":Cvec,
              "gamma":gammavec}

gs_gaussian = GridSearchCV(SVC(kernel='rbf'),parameters)
pip_gaussian = make_pipeline(StandardScaler(),gs_gaussian)
n = 10000
Xa,ya=gethomogeneus_datast(Xtrain_1d,y_train,10,n)

np.histogram(ya)

pip_gaussian.fit(Xa,ya.ravel())

bestC = gs_gaussian.best_params_['C']
bestgamma = gs_gaussian.best_params_['gamma']
print(bestgamma)
print(bestC)
