import numpy as np
import skimage
import matplotlib.pylab as plt
from skimage import data
from skimage.util import img_as_float
import wavelet_transform


shrink = (slice(0, None, 1), slice(0, None, 1))
image = img_as_float(data.load('brick.png'))[shrink]
image = image[0:64,0:64]

J = 4  # number of scales
L = 8  # number of angles per scale

# Wavelet transform:
Filters = filter_bank_morlet2d(J=4,L=8)
x_filtered=apply_conv(image,Filters,mode='wrap')

# and plot
plt.figure(figsize=(16,8))
for j,list_filters in enumerate(Filters['psi']):
    for l,filter in enumerate(Filters['psi'][j]):
        plt.subplot(J,L,j*L+l+1)
       # filtered = np.real(ndi.convolve(image, filter, mode='wrap'))
        filtered = x_filtered['psi'][j][l]
        plt.imshow(np.real(filtered),interpolation='nearest')
        plt.title('si-images')
        plt.viridis()


####################
"""  # TRANSLATION
        sigma = sigma_xi*scale
        xi = xi_psi/scale
        slant = 5./L


        # ivans+michael's code
        filter = gabor_2d((2**J,2**J), sigma, xi, theta, slant=slant)

        #EXACT TRANSLATION! OF the paramenters between the two notations
        sigma_x = sigma
        sigma_y = sigma/slant
        freq = xi/(np.pi*2) #xi_psi/(2*np.pi*scale)

        # scikit-image gabor
        gabor = gabor_kernel(freq,theta=theta,sigma_x=sigma_x,sigma_y=sigma_y)


        """