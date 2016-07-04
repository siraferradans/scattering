import os, sys
import scipy.misc
import scipy.io
import glob
import time as time
from skimage import img_as_float
#Load data: extract patches and label per patch
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import data
import numpy as np
from scattering.filter_bank import filter_bank_morlet2d
from scattering.scattering import scattering


def generate_scattering(images):
    
    N = images.shape[1]
    L=8
    J = np.log2(N)
    wavelet_filters,littlewood = filter_bank_morlet2d(N,J=J,L=L)
    
    S,U = scattering(images,wavelet_filters)
    
    return S


def load_image_patches_curet(dataset_path,px = 128):

    curet_patches = []
    curet_labels = []
    i = -1
    print('Loading images (in BW!!):')
    t_images = time.time()
    for folder, sub_folders, files in os.walk(dataset_path):
        print('i=', i + 1)
        i += 1
        for file in files:
            # exit  print('file:',file)
            if file.endswith(".png"):
                file_path = os.path.join(folder, file)

                patches = extract_patches_2d(img_as_float(data.imread(file_path)), [px, px], max_patches=4)
                # getting black and white images
                # patches = patches.transpose((0,3,1,2))
                # patches = patches.reshape((patches.shape[0]*patches.shape[1],px,px))
                patches = patches[:, :, :, 0]
                curet_patches.append(patches)
                curet_labels.append([i] * len(curet_patches[-1]))

    curet_labels = np.concatenate(curet_labels, axis=0)
    curet_patches = np.concatenate(curet_patches, axis=0)
    print(curet_patches.shape[0], ' images loaded in ', time.time() - t_images, ' secs')

    print('shape data:', curet_patches.shape)
    print('Compute scatterings:')
    t_scats = time.time()
    scatterings=[]
    for i in np.arange(0,curet_patches.shape[0],30):
        print(i,'/',curet_patches.shape[0])
        scatterings.append(generate_scattering(curet_patches[i:i+29, :, :]))

    scatterings = np.concatenate(scatterings, axis=0)

    print('saving data in ./curet.mat')
    scipy.io.savemat('./curet.mat', mdict={'labels': curet_labels, 'scatterings':scatterings})
    print('saving done!')

    print(scatterings.shape[0], ' scat. features computed in ', t_scats - time.time(), ' secs.')

    return curet_labels, scatterings

