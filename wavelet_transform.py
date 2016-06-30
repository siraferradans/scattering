import numpy as np
import skimage
from skimage.filters import gabor_kernel
from skimage import data
from scipy import ndimage as ndi


def morlet_2d(sigma, xi, theta, slant=0.5):
#spatial domain (non-Fourier)

    # TRANSLATION
    #exact translation of the paramenters between the two notations
    sigma_x = sigma
    sigma_y = sigma/slant
    freq = xi/(np.pi*2)

    gabor = gabor_kernel(freq,theta=theta,sigma_x=sigma_x,sigma_y=sigma_y)

    #gabor = gabor_2d(shape,sigma, xi, theta, slant)
    envelope = np.abs(gabor)
    K = gabor.sum() / envelope.sum()

    centered = gabor - K * envelope

    return centered


def filter_bank_morlet2d(J=4, L=8, filter_type='morlet_2d',
                         conv_type='spatial', bb='reflect'):
    # IVANS format: filters = dict(phi=lowpass_fourier, psi=dict(filt_list=filt_list),
    #                   lam=lambda_list, J=J, L=L, Q=Q)


    # TODO: boundary values, and conv_type de fourier
    # tambien controlar que solo se puede meter morlet_2d

    ## phi: Create low pass filter
    # max scale (lower freq)
    max_scale = 2 ** (float(J - 1))  # again Q=1
    sigma_phi = .8
    sigma = sigma_phi * max_scale
    xi_phi = 0
    freq = xi_phi / max_scale

    phi_filter = gabor_kernel(freq, theta=0, sigma_x=sigma, sigma_y=sigma)

    ## psi: Create band-pass filters
    # constant values for psi
    xi_psi = 3. / 4 * np.pi  # for Q=1
    sigma_xi = .8
    slant = 4. / L

    filters_psi = []
    for j, scale in enumerate(2 ** np.arange(J)):
        angle_list = []
        for l, theta in enumerate(np.arange(L) / float(L) * np.pi):
            sigma = sigma_xi * scale
            xi = xi_psi / scale

            psi = morlet_2d(sigma, xi, theta, slant=slant)
            angle_list.append(psi)

        filters_psi.append(angle_list)

    # we access the Filters asL Filters['phi'][scale][angle]
    Filters = dict(phi=phi_filter, psi=filters_psi)

    return Filters


def apply_conv(x,Filters,mode='wrap'):

    phi = Filters['phi']
    x_phi_filtered = ndi.convolve(x, phi, mode=mode)

    x_psi_filtered = []
    for j,list_filters in enumerate(Filters['psi']):
        x_psi_l = []
        for l,filter in enumerate(list_filters):
            x_psi_l.append(np.real(ndi.convolve(x, filter, mode=mode)))

        x_psi_filtered.append(x_psi_l)

    x_filtered = dict(phi = x_phi_filtered, psi=x_psi_filtered)

    return x_filtered



