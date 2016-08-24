
##### Gaussian kernel grid search
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import h5py
import numpy as np



def gethomogeneus_datast(X, y, d, n):
    num_per_class = np.int(n / d)
    ytrain = np.reshape(y, (y.shape[0],))

    X_out = []  # np.zeros(n,X.shape[1],X.shape[2],X.shape[3])
    y_out = []
    for i_d in np.arange(d):
        indx = np.where(ytrain.ravel() == i_d)

        X_out.append(X[indx[0][0:num_per_class], :])
        y_out.append(ytrain[indx[0][0:num_per_class],])

    X_out = np.concatenate(X_out, axis=0)
    y_out = np.concatenate(y_out, axis=0)
    # print(X_out.shape)
    # print(y_out)
    return X_out, y_out

######### Load data
import numpy as np
n = 60000
#from load_cifar10_db import load_images_cifar,load_scattering_cifar
#X_train, y_train, X_test, y_test = load_scattering_cifar(num_images = 150000, J=3,L=8,m=1, sigma_phi=0.8, sigma_xi=0.8)
#

#Load the saved data
f = h5py.File('./cifar10_scatteringdata.mat')
X_train = np.array(f['X_train']).astype('single')
y_train = np.array(f['y_train']).astype('single')
X_test = np.array(f['X_test']).astype('single')
y_test = np.array(f['y_test']).astype('single')
m = np.array(f['m'])
J = np.array(f['J'])
L = np.array(f['L'])

num_samples = X_train.shape[0]


X_train.shape
Xtrain_1d = X_train.reshape((50000,75*8*8))
Xtest_1d = X_test.reshape((10000,75*8*8))


##### Gaussian kernel evaluate
bestC = 3.8
bestgamma = 10**(-3.6)

gs_gaussian = SVC(kernel='rbf', C=bestC, gamma=bestgamma)
pip_gaussian = make_pipeline(StandardScaler(),gs_gaussian)
n = 40000
Xa,ya=gethomogeneus_datast(Xtrain_1d,y_train,10,n)

start = time.time()
pip_gaussian.fit(Xa,ya.ravel())

print(np.log10(bestgamma))
print(bestC)
out=pip_gaussian.predict(Xtest_1d)

score = accuracy_score(y_test, out)
print(score) # gamma = -3.5, C=4.1, score=0.4179, tol = 0.1 , time: 45.95646786689758,n=300
print('time:',time.time()-start, ' for n=',n)

print('vs',
"""
n=1000, tol = 0.1
-3.6
3.8
0.5169
time: 387.4123179912567
""")