import numpy as np
import skimage
from skimage._shared.utils import assert_nD
from scipy import ndimage as ndi
from scattering.filter_bank import filter_bank_morlet2d, filterbank_to_multiresolutionfilterbank
import warnings

def _apply_fourier_mult(signals,filters):
# Assume signals and filters are in the Fourier domain
# and the dimensions are:
# - signals: (num_signals, Nj,Nj) (color images can be stacked in the num_signals)
# - filters: [L,N,N] where L is the num. of filters to apply

    N = filters.shape[1]
    Nj= signals.shape[1]

    filtered_signals=signals[:,np.newaxis,:,:] * filters[np.newaxis,:,:,:]
    return filtered_signals


def _subsample(X,j):
    """Spatial subsampling on the last two dimensions of X at a rate that depends on 2**j.

    Parameters
    ----------
    X : array like variable.
        3D dnarray with shape (N,L,n,n) or (N,n,n). The subsampling is produced in the last two
        dimensions (n,n)
    j : int
        Rate of subsampling is 2**j

    Returns
    -------
    XX : ndarray
        Subsampled images
    """

    dsf = 2**j
    
    return dsf*X[..., ::dsf, ::dsf]


def _apply_lowpass(img, phi, J, n_scat):
    """Applies a low pass filter on the 'img' set of 2D arrays and subsamples.

    Convolution the filter phi (in the Fourier domain) on the set of images defined by img. The convolution is done in
    the Fourier domain, thus it is a pointwise multiplication of the filter phi and the images, stored in
    the last 2 dimensions of img. Then, the images are subsampled at the appropiate rate to have 'n_scat'
    spatial coefficients

    Parameters
    ----------
    img : ndarray of (N,L,n,n) or (N,n,n) shape.
        Input images in the spatial domain, stacked along the first dimensions.
    phi : Low pass filter as a 2D ndarray
        Low pass filter in the Fourier domain, stored as (n,n) matrix
    J : int
        Rate of subsampling 2**J
    n_scat : int
        number of spatial coefficients of the scattering vector

    Returns
    -------
    XX : ndarray
        stacked images after being filtered and subsampled
    """

    img_filtered = np.real(np.fft.ifft2(np.fft.fft2(img)*phi))

    n_nolp = img.shape[-1]

    ds = int(n_nolp / n_scat)
    
    return 2 ** (J - 1) * img_filtered[..., ::ds, ::ds]


def scattering(x,wavelet_filters=None,m=2):
    """Compute the scattering transform of a signal (or set of signals).

    Given 'x', a set of 2D signals, this function computes the scattering transform
    of these signals using the filter bank 'wavelet_filters'.

    Notes
    -----
    **Bondary values: The scattering transform applies a set of convolutions to the input signals.
    These convolutions are computed as the point-wise multiplication in the Fourier domain, thus
    the boundary values of the image are circular or cyclic. In case you need other kind of boundary
    values, for instance zero-padded, you should edit the images before calling this function.

    **Shape of x: The signals x must be squared shaped, thus (N,px,px) and not (N,px,py) for py != px. In case
    the images are rectangular they will be cropped to the smallest dimension, px or py.

    Parameters
    ----------
    x : array_like
        3D dnarray with N images (2D arrays) of size (px,px), thus x has size (N,px,px)
        In case the array is rectangular (N,px,py) for px not equal to py, the images will be cropped.
    wavelet_filters : Dictionary with the multiresolution wavelet filter bank
        Dictionary of vectors obtained after calling:
            >>>> px = 32 #number of pixels of the images
            >>>> J = np.log2(px) #number of scales
            >>>> wf, l = filter_bank_morlet2d(px,J=J)
            >>>> wavelet_filters = filterbank_to_multiresolutionfilterbank(wf,J)
    m : int
        Order of the scattering transform, which can be 0, 1 or 2.


    Returns
    -------
    S : array_like
        Scattering transform of the x signals
    U : array_like
        Result before applying the lowpass filter and subsampling 

    Raises
    ------
    UserWarning
        If the size of x is not (N,px,px) and informs that the images with be cropped to (N,px,px)
    UserWarning
        If no wavelet filters, the function creates a multiresolution set of filters with predefined
        settings, but warns about the parameters and suggests precomputing the filters.
    UserWarning
        If the value of m is not 0,1, or 2.

    Examples
    --------
    Processing a set of 3 (randomly generated) images:

            >>>> import numpy as np
            >>>> from scattering.filter_bank import filter_bank_morlet2d, filterbank_to_multiresolutionfilterbank
            >>>> from scattering.scattering import scattering

            >>>> px = 32 #number of pixels of the images
            >>>> images = np.arange(0,3*px*px).reshape(3,px,px) #create images
            >>>> wf, l = filter_bank_morlet2d(px,J=np.log2(px))
            >>>> wavelet_filters = filterbank_to_multiresolutionfilterbank(wf,J)
            >>>> S,U = scattering(images,m=2)
    """

    ## Check that the input data is correct:
    ## 1.- Signals are squared, otherwise, crop
    assert_nD(x, 3, 'x')  # check that the images are correctly stacked
    num_signals, px, py = x.shape

    if (px != py):
        warning_string = "Variable x has shape {0}, which is not in format (N,px,px). We crop to the smallest dimension."
        warnings.warn(warning_string.format(x.shape))
        px = min(px,py)
        x = x[:, 0:px, 0:px]

    ## 2.- If we dont have filters, get them with the defailt values, J=3, L=8
    if wavelet_filters is None:
        J = int(min(np.log2(px),3))
        L = 8
        warning_string = "No filter input, we create a Morlet filter bank with J= {0} and L={1}. Strongly suggest creating " \
                         "the filters before hand and pass them as a parameter."
        warnings.warn(warning_string.format(J,L))
        wf, littlewood = filter_bank_morlet2d(px, J=J, L=L)
        wavelet_filters = filterbank_to_multiresolutionfilterbank(wf, J)
    else:
        J = len(wavelet_filters['psi'][0])  # number of scales
        L = len(wavelet_filters['psi'][0][0])  # number of orientations

    num_signals = x.shape[0]

    ## 3.- Check the Order of the scattering transform, can only be 0,1,2, and that gives us different
    # number of scattering coefs
    num_coefs = {
        0: int(1),
        1: int(1 + J * L),
        2: int(1 + J * L + J * (J - 1) * L ** 2 / 2)
    }.get(m, -1)

    if num_coefs == -1:
        warning_string = "Parameter m out of bounds, valid values are 0,1,2 not {0}"
        warnings.warn(warning_string.format(m))
        return

    spatial_coefs = int(x.shape[1]/2**(J-1))

    oversample = 1  # subsample at a rate a bit lower than the critic frequency

    U = []
    V = []
    v_resolution = []
    S = np.ndarray((num_signals,num_coefs,spatial_coefs,spatial_coefs))

    current_resolution = 0

    #Zero order scattering coeffs
    S[:, 0, :, :] = _apply_lowpass(x, wavelet_filters['phi'][current_resolution], J,  spatial_coefs)

    #First order scattering coeffs
    if m>0:
        Sview = S[:,1:J*L+1,:,:].view()
        Sview.shape=(num_signals,J,L,spatial_coefs,spatial_coefs)

        X = np.fft.fft2(x) # precompute the fourier transform of the images
        for j in np.arange(J):
            filtersj = wavelet_filters['psi'][current_resolution][j].view()

            resolution = max(j-oversample, 0)# resolution for the next layer
            v_resolution.append(resolution)

            # fft2(| x conv Psi_j |): X is full resolution, as well as the filters
            aux = _subsample(np.fft.ifft2(_apply_fourier_mult(X,filtersj)), resolution )

            V.append( aux )
            U.append( np.abs(aux))

            Sview[:, j, :, :, :] = _apply_lowpass(U[j], wavelet_filters['phi'][resolution], J,  spatial_coefs)

    #Second order scattering coeffs
    if m>1:
        sec_order_coefs = int(J*(J-1)*L**2/2)
        
        S2norder = S[:,J*L+1:num_coefs,:,:] # view of the data
        S2norder.shape = (num_signals, int(sec_order_coefs/L), L, spatial_coefs, spatial_coefs)
        
        indx = 0
        for j1 in np.arange(J):
            Uj1 = np.fft.fft2(U[j1].view()) # U is in the spatial domain
            current_resolution = v_resolution[j1]
            
            for l1 in np.arange(Uj1.shape[1]):
                Ujl1 = Uj1[:,l1,].view() # all images single angle, all spatial coefficients
                
                for j2 in np.arange(j1+1,J):
                    # | U_lambda1 conv Psi_lambda2 | conv phi
                    aux = np.abs(np.fft.ifft2(_apply_fourier_mult(Ujl1, wavelet_filters['psi'][current_resolution][j2])))
                    # computing all angles at once
                    S2norder[:,indx,:,:,:]= \
                        _apply_lowpass(aux, wavelet_filters['phi'][current_resolution], J,  spatial_coefs)
                    
                    indx = indx+1
    
#Rototranslation should use the V

    return S,U

"""
def apply_lowpass_fast(img, phi, J, N_scat):

    #This function can be applied to a set of images and angles, the input img is assumed
    #to be of size (num_images,L,size_x,size_y)

    N_nolp = img.shape[-1]

    N_nolp_r = N_nolp // 2 + 1
    # dsf = N_nolp / N_scat
    dsf = N_scat
    out = np.fft.rfft(img, axis=-1) * phi[0, :N_nolp_r].reshape(1, N_nolp_r)
    out = np.fft.irfft(out, axis=-1)[..., ::dsf].copy()

    out = np.fft.rfft(out, axis=-2) * phi[:N_nolp_r, 0].reshape(N_nolp_r, 1)
    out = np.fft.irfft(out, axis=-2)[..., ::dsf, :]
    out = 2 ** (J - 1) * np.real(out)

    return out



def apply_lowpass_ivan(img, phi, J, N_scat,type_complex='complex64'):
    
    #fft, ifft, fft2, ifft2, rfft, irfft, rfft2, irfft2 = select_fft(fft_choice)
    
    fft_module = np.fft
    
    fft = fft_module.fft
    ifft = fft_module.ifft
    fft2 = fft_module.fft2
    ifft2 = fft_module.ifft2
    rfft = fft_module.rfft
    irfft = fft_module.irfft
    
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
    
    #    if fft_choice == 'mkl_fft':
    #    out = np.fft.rfft(img, axis=1, out=out) * phi[0, :N_nolp_r].reshape(1, N_nolp_r)
    #else:
    out[:] = rfft(img, axis=1) * phi[0, :N_nolp_r].reshape(1, N_nolp_r)
    
    out = irfft(out, axis=1)[:, ::dsf].copy()
    out = rfft(out, axis=0) * phi[:N_nolp_r, 0].reshape(N_nolp_r, 1)
    out = irfft(out, axis=0)[::dsf, :]
    out = 2**(J - 1) * np.real(out)
    
    return out
"""
