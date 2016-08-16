import os, sys
import scipy.misc
import scipy.io
import glob
import time as time


#from skimage import img_as_float
# Load data: extract patches and label per patch
from sklearn.feature_extraction.image import extract_patches_2d
#from skimage import data
from skimage.color.colorconv import rgb2yuv,yuv2rgb
import skimage.transform as sct

from keras.datasets import cifar10
import numpy as np
from scattering.filter_bank import filter_bank_morlet2d
from scattering.scattering import scattering


def DB_rgb2yuv(X):
    num_samples,c,px,px = X.shape
    Xta = X.transpose((3,2,0,1))/255
    Iyuv = rgb2yuv(Xta).transpose((2,3,1,0)).copy()
    Iyuv.shape = (num_samples*3,px,px)
    return Iyuv

def load_images_cifar():
    # the data, shuffled and split between train and test sets
    px=32
    # input image dimensions
    img_rows, img_cols = px, px

    (X_train_sm, y_train), (X_test_sm, y_test) = cifar10.load_data()
    num_images_ta = X_train_sm.shape[0]
    num_images_te = X_test_sm.shape[0]

    #need to change to YUV
    X_train = DB_rgb2yuv(X_train_sm.astype('float32'))
    X_test =  DB_rgb2yuv(X_test_sm.astype('float32'))
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, y_train, X_test, y_test


def load_scattering_cifar(num_images = 300, J=3,L=8,m=2):

    i = -1
    epsilon = 1e-6
    print('Loading images:')
    t_images = time.time()

    X_train, y_train, X_test, y_test = load_images_cifar()

    px = X_train.shape[-1]
    print(X_train.shape[0], ' images loaded in ', time.time() - t_images, ' secs')
    print('shape data:', X_train.shape)

    print('Create filters:')
    
    wavelet_filters, littlewood = filter_bank_morlet2d(px, J=J, L=L, sigma_phi=0.6957,sigma_xi=0.8506 )

    ### Generate Training set
    print('Compute ', num_images, ' scatterings:')
    t_scats = time.time()
    scatterings_train = []
    scatterings_test =[]
    step = 600
    for i in np.arange(0, min(num_images*3, X_train.shape[0]), step):
        print(i, '/', min(num_images, X_train.shape[0]))
        S,u = scattering(X_train[i:i + step, :, :], wavelet_filters,m=m)

        scatterings_train.append(np.log(np.abs(S)+epsilon))

    scatterings_train = np.concatenate(scatterings_train, axis=0)

    if (np.isnan(np.sum(scatterings_train[:]))):
        print('Error: we have a nans in the training set')

    #putting color channels together
    num_files,scat_coefs,spatial,spatial = scatterings_train.shape
    scatterings_train.shape = (num_files/3,3*scat_coefs,spatial,spatial)

    print(scatterings_train.shape[0], ' scat. features computed in ', time.time() - t_scats, ' secs.')

    ### Generate Testing set
    print('Now testing set:')

    t_scats = time.time()
    for i in np.arange(0, X_test.shape[0], step):
        print(i, '/', min(num_images*3, X_test.shape[0]))
        S,u = scattering(X_test[i:i + step , :, :], wavelet_filters,m=m)

        scatterings_test.append(np.log(np.abs(S)+epsilon))

    scatterings_test = np.concatenate(scatterings_test, axis=0)

    if (np.isnan(np.sum(scatterings_test[:]))):
        print('Error: we have a nans in the test set')

    #putting color channels together
    num_files,scat_coefs,spatial,spatial = scatterings_test.shape
    scatterings_test.shape = (num_files/3,3*scat_coefs,spatial,spatial)


    print(scatterings_test.shape[0], ' scat. 3-color features computed in ', time.time() - t_scats , ' secs.')

    # print('saving data in ./mnist_scat.mat')
    #scipy.io.savemat('./mnist_scat.mat', mdict={'ytrain': y_train, 'xtrain': scatterings_train, 'ytest': y_test, 'xtest': scatterings_test})
    #print('saving done!')

    return scatterings_train, y_train[0:scatterings_train.shape[0]], scatterings_test, y_test[0:scatterings_test.shape[0]]
