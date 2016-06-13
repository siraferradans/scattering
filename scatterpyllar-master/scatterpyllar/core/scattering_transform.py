# import pyfftw
import numpy as np
import scipy.special 
import scipy.fftpack
import sys
import time

import mkl_fft

from ..filters.utils import fft_convolve


def generate_lambda_list(J, L, max_layer=2):
    """Generate a list of lambda indices of the scattering transform.

    This is mainly used in gradient computations for inverse scattering.

    Parameters
    ----------
    J : int
        The largest wavelet scale used.
    L : int
        Number of rotations per scale.
    max_layer : int, optional
        The number of layers. By default, and in practice always, 2.

    Returns
    -------
    lambda_list : list 

        A list of length max_layer with the i-th entry being the list of
        lambda indices at the i-th layer. The indices themselves are tuples of
        tuples so that the parent indices are included.

    Examples
    --------
    >>> generate_lambda_list(2, 2)

    """

    lambda_list = [[((0, 0),)]]
    for layer_m in range(max_layer):
        for lam in lambda_list[layer_m]:
            print lam
            lam_children = [lam + ((j, l),) for j in range(lam[-1][0] + 1, J + 1)
                            for l in range(L)]
            
            # if there are any children, add them to the list
            if lam_children:
                lambda_list.append(lam_children)


    return lambda_list




def number_of_coeffs(J, L, max_layer=2):
    """Returns the total number of coefficients in a delocalized scattering
    transform (the transform size).

    Parameters
    ----------
    J : int
        The largest wavelet scale used.
    L : int
        Number of rotations per scale.
    max_layer : int, optional
        The number of layers. By default, and in practice always, 2.

    Returns
    -------
    n : int
        The number of coefficients.

    """

    # A subsampled image is considered to be 1 coefficient. The actual number
    # of scalars is larger.

    n_coeffs = scipy.special.binom(J, 
        np.arange(max_layer + 1)) * (L ** np.arange(max_layer + 1))
    return n_coeffs.sum()


def scat2vec(scat):
    """Vectorize a delocalized scattering dictionary.

    Parameters
    ----------
    scat : dict
        The scattering transform dictionary.

    Returns
    -------
    vec : array_like
        An (n_coeffs by 1) vector of scattering coefficients.

    See Also
    --------
    vec2scat


    """


    vec = np.zeros(
        (len(scat['coeffs']), 1),
        dtype=scat['coeffs'][(0, 0), ]['l1'].dtype
    )

    # TO DO: change the scattering structure to make the following faster
    # e.g. scat['l1'][:] and such...

    i = 0
    for layer_m in range(len(scat['lambda_list'])):
        for lam in scat['lambda_list'][layer_m]:
            vec[i] = scat['coeffs'][lam]['l1'][0]
            i += 1

    return vec


def vec2scat(vec, scat):
    """Convert a scattering vector back to a dictionary.

    Parameters
    ----------
    vec : array_like
        An (n_coeffs by 1) vector of scattering coefficients.
    scat : dict
        The scattering transform dictionary.

    Returns
    -------
    scat : dict
        The scattering transform dictionary.

    See Also
    --------
    scat2vec

    Notes
    -----
    Conversion is in-place, so that the returned object is same as scat.
    

    """

    i = 0
    for layer_m in range(len(scat['lambda_list'])):
        for lam in scat['lambda_list'][layer_m]:
            # this changes "scat" in place

            scat['coeffs'][lam]['l1'][0] = vec[i, 0]
            i += 1

    return scat


def select_fft(fft_choice):
    if fft_choice == 'fftw':
        fft_module =  pyfftw.interfaces.numpy_fft
    elif fft_choice == 'fftpack':
        # Fortran FFTPACK from scipy
        fft_module = scipy.fftpack
    elif fft_choice == 'fftpack_lite':
        # C FFTPACK light from numpy
        fft_module = np.fft
    elif fft_choice == 'mkl_fft':
        fft_module = mkl_fft

    else:
        raise ValueError('Non-existing FFT library requested.')
    
    fft = fft_module.fft
    ifft = fft_module.ifft
    fft2 = fft_module.fft2
    ifft2 = fft_module.ifft2
    rfft = fft_module.rfft
    irfft = fft_module.irfft


    return fft, ifft, fft2, ifft2, rfft, irfft


def scattering_transform(img, filter_bank, localized=True, dtype='single', fft_choice='mkl_fft'):

    max_layer = 2

    fft, ifft, fft2, ifft2, rfft, irfft = select_fft(fft_choice)

    N = img.shape[0]
    logN = np.log2(N)
    J = filter_bank['J']
    L = filter_bank['L']

    N_scat = 2 ** (logN - (J - 1))  # rows/cols per coefficient
    if localized is True:
        scat_shape = N_scat, N_scat
    else:
        scat_shape = 1,

    lambda_list = [[((0, 0),)]]     # list of lambdas per layer

    if dtype is 'single':
        dtype_complex = 'complex64'
    else:
        dtype_complex = 'complex128'

    ## Allocate the transform container
    value = np.zeros((number_of_coeffs(J, L, max_layer),) + scat_shape, \
        dtype=dtype)
    scat = dict(coeffs=dict(), all_values=value)

    # TO DO: if you add l2, still all the pointers can be nicely done. The
    # corresponding index in value should go between lambda and shape

    # TO DO: Does it make sense to store abs? Do we get any speedup?
    # In any case, signal and envelope should be contiguous... TO DO!
    # If you do that, do it also in the gradients.

    value_no_lowpass = dict()
    value_no_lowpass[lambda_list[0][0]] = dict()
    value_no_lowpass[lambda_list[0][0]]['signal'] = img.copy()
    value_no_lowpass[lambda_list[0][0]]['envelope'] = img

    # Enable FFTW caching. TO DO: Byte-align arrays to get more speed
    # pyfftw.interfaces.cache.enable()

    i_coeff = 0
    for layer_m in range(max_layer + 1):
        if layer_m < max_layer:
            lambda_list.append([])

        for lam in lambda_list[layer_m]:

            ## Output the scattering coefficients at the current layer
            scat['coeffs'][lam] = dict()
            scat['coeffs'][lam]['l1'] = value[i_coeff]
            N_nolp = value_no_lowpass[lam]['envelope'].shape[0]
            res = int(np.log2(N / N_nolp))


            # TO DO: write nice functions for all of this
            if localized:
                res_phi = int(np.log2(N / N_nolp))
                phi = filter_bank['phi'][res_phi]
                scat['coeffs'][lam]['l1'][:] = \
                    apply_lowpass(value_no_lowpass[lam]['envelope'], phi, J, N_scat, fft, ifft)
            else:
                scat['coeffs'][lam]['l1'][:] = \
                    value_no_lowpass[lam]['envelope'].mean()


            ## Compute the convolutions for the next layer
            if layer_m < max_layer:
                F_U_ml = mkl_fft.cce2full(mkl_fft.mkl_rfft2(value_no_lowpass[lam]['envelope']))
                
                lam_children = [(j, l) for j in range(lam[-1][0] + 1, J + 1) 
                                       for l in range(L)]

                for lam_child in lam_children:
                    lam_total = lam + (lam_child,)
                    lambda_list[layer_m + 1].append(lam_total)
                    scale = lam_child[0]

                    dsf = N_nolp / (N / 2**(scale - 1))

                    # NB: this will not be contiguous in memory. TO DO:
                    # implement a contiguous variant and test which one is
                    # faster (also in inversion):
                    full_conv = ifft2(F_U_ml * filter_bank['psi'][(lam_child, res)])

                    value_no_lowpass[lam_total] = dict()
                    value_no_lowpass[lam_total]['signal'] = \
                        full_conv[::dsf, ::dsf].copy()
                    value_no_lowpass[lam_total]['envelope'] = \
                        np.abs(value_no_lowpass[lam_total]['signal'])


            i_coeff += 1

    scat['lambda_list'] = lambda_list
    return scat, value_no_lowpass


def apply_lowpass(img, phi, J, N_scat, fft, ifft, rfft, irfft, type_complex='complex64'):

    # NB: I could compute N_scat here, but in case we want to oversample, this
    # may be useful. I should make a class.

    N_nolp = img.shape[0]
    N_nolp_r = N_nolp//2 + 1
    dsf = N_nolp / N_scat

    # out will in the end point to a downsampled lowpassed image
    # out = ifft(fft(img, axis=1)
    #    * phi[0, :].reshape(1, N_nolp), axis=1)[:, ::dsf]

    # TO DO: use real transforms

    out = np.zeros((img.shape[0], N_nolp_r), dtype=type_complex)
    out = rfft(img, axis=1, out=out) * phi[0, :N_nolp_r].reshape(1, N_nolp_r)
    out = irfft(out, axis=1)[:, ::dsf].copy()
    out = rfft(out, axis=0) * phi[:N_nolp_r, 0].reshape(N_nolp_r, 1)
    out = irfft(out, axis=0)[::dsf, :]
    out = 2**(J - 1) * np.real(out)

    return out

    # out = ifft(fft(out, axis=0)
    #    * phi[:, 0].reshape(N_nolp, 1), axis=0)[::dsf, :]
