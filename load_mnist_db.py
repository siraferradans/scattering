import os, sys
import scipy.misc
import scipy.io
import glob
import time as time
#from skimage import img_as_float
# Load data: extract patches and label per patch
from sklearn.feature_extraction.image import extract_patches_2d
#from skimage import data

import skimage.transform as sct

from keras.datasets import mnist
import numpy as np
from scattering.filter_bank import filter_bank_morlet2d, filterbank_to_multiresolutionfilterbank

from scattering.scattering import scattering


def load_images_mnist(px=32):
    # the data, shuffled and split between train and test sets

    # input image dimensions
    img_rows, img_cols = px, px

    (X_train_sm, y_train), (X_test_sm, y_test) = mnist.load_data()
    num_images_ta = X_train_sm.shape[0]
    num_images_te = X_test_sm.shape[0]

    X_train = np.zeros((num_images_ta,px, px))
    X_train[:, 3:31, 3:31] = X_train_sm

    X_test = np.zeros((num_images_te, img_rows, img_cols))
    X_test[:, 3:31, 3:31] = X_test_sm
    #X_train = sct.resize(X_train.transpose((1, 2, 0)), (px, px)).transpose((2, 0, 1))
    #X_test  = sct.resize(X_test.transpose((1, 2, 0)), (px, px)).transpose((2, 0, 1))

#    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    return X_train, y_train, X_test, y_test


def load_scattering_mnist(num_images = 300, px=32,J=3,L=8,m=2):

    i = -1
    print('Loading images (in BW!!):')
    t_images = time.time()

    X_train, y_train, X_test, y_test = load_images_mnist(px=px)


    print(X_train.shape[0], ' images loaded in ', time.time() - t_images, ' secs')
    print('shape data:', X_train.shape)

    print('Create filters:')
    
    wavelet_filters, littlewood = filter_bank_morlet2d(px, J=J, L=L, sigma_phi=0.6957,sigma_xi=0.8506 )

    ### Generate Training set
    print('Compute ', num_images, ' scatterings:')
    t_scats = time.time()
    scatterings_train = []
    scatterings_test =[]
    step = 500
    for i in np.arange(0, min(num_images, X_train.shape[0]), step):
        print(i, '/', min(num_images, X_train.shape[0]))
        S,u = scattering(X_train[i:i + step, :, :], wavelet_filters,m=m)
        scatterings_train.append(S)

    scatterings_train = np.concatenate(scatterings_train, axis=0)

    print(scatterings_train.shape[0], ' scat. features computed in ', time.time() - t_scats , ' secs.')

    ### Generate Testing set
    print('Now testing set:')

    t_scats = time.time()
    for i in np.arange(0, X_test.shape[0], step):
        print(i, '/', min(num_images, X_test.shape[0]))
        S,u = scattering(X_test[i:i + step , :, :], wavelet_filters,m=m)
        scatterings_test.append(S)

    scatterings_test = np.concatenate(scatterings_test, axis=0)
    print(scatterings_test.shape[0], ' scat. features computed in ', time.time() - t_scats , ' secs.')

    # print('saving data in ./mnist_scat.mat')
    #scipy.io.savemat('./mnist_scat.mat', mdict={'ytrain': y_train, 'xtrain': scatterings_train, 'ytest': y_test, 'xtest': scatterings_test})
    #print('saving done!')

    return scatterings_train, y_train[0:scatterings_train.shape[0]], scatterings_test, y_test[0:scatterings_test.shape[0]]


def load_scattering_multiresolution_mnist(num_images = 300, px=32,J=3,L=8,m=2):

    i = -1
    print('Loading images (in BW!!):')
    t_images = time.time()

    X_train, y_train, X_test, y_test = load_images_mnist(px=px)


    print(X_train.shape[0], ' images loaded in ', time.time() - t_images, ' secs')
    print('shape data:', X_train.shape)

    print('Create filters:')
    
    wavelet_filters, littlewood = filter_bank_morlet2d(px, J=J, L=L, sigma_phi=0.6957,sigma_xi=0.8506 )

    Filters = filterbank_to_multiresolutionfilterbank(wavelet_filters,J)
  
    ### Generate Training set
    print('Compute ', num_images, ' scatterings:')
    t_scats = time.time()
    scatterings_train = []
    scatterings_test =[]
    step = 500
    for i in np.arange(0, min(num_images, X_train.shape[0]), step):
        print(i, '/', min(num_images, X_train.shape[0]))
        S,u = scattering(X_train[i:i + step, :, :], Filters,m=m)
        scatterings_train.append(S)

    scatterings_train = np.concatenate(scatterings_train, axis=0)

    print(scatterings_train.shape[0], ' scat. features computed in ', time.time() - t_scats , ' secs.')

    ### Generate Testing set
    print('Now testing set:')

    t_scats = time.time()
    for i in np.arange(0, X_test.shape[0], step):
        print(i, '/', min(num_images, X_test.shape[0]))
        S,u = scattering(X_test[i:i + step , :, :], Filters,m=m)
        scatterings_test.append(S)

    scatterings_test = np.concatenate(scatterings_test, axis=0)
    print(scatterings_test.shape[0], ' scat. features computed in ', time.time() - t_scats , ' secs.')

    # print('saving data in ./mnist_scat.mat')
    #scipy.io.savemat('./mnist_scat.mat', mdict={'ytrain': y_train, 'xtrain': scatterings_train, 'ytest': y_test, 'xtest': scatterings_test})
    #print('saving done!')

    return scatterings_train, y_train[0:scatterings_train.shape[0]], scatterings_test, y_test[0:scatterings_test.shape[0]]
