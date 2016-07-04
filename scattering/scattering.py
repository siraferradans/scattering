import numpy as np
import skimage
from scipy import ndimage as ndi


def apply_fourier_mult(signals,filters):
# Assume signals and filters are in the Fourier domain
# and the dimensions are:
# - signals: (num_signals, N,N) (color images can be stacked in the num_signals)
# - filters: [L,N,N] where L is the num. of filters to apply

# Goal pointwise multiplication

    L = filters.shape[0]
    N = filters.shape[1]
    num_samples = signals.shape[0]

  #  filtered_signals = np.ndarray(num_samples,L,N,N)
    #TODO: SELECT THE CORRECT AMOUNT ACCORDING TO THE SUBSAMPLING, and control the size

    filtered_signals=signals[:,np.newaxis,:,:] * filters[np.newaxis,:,:,:]

    return filtered_signals


def apply_lowpass(img, phi, J, N_scat):
    # NB: I could compute N_scat here, but in case we want to oversample, this
    # may be useful. I should make a class.

    #This function can be applied to a set of images and angles, the input img is assumed
    #to be of size (num_images,L,size_x,size_y)

    N_nolp = img.shape[-1]

    N_nolp_r = N_nolp // 2 + 1
    dsf = N_nolp / N_scat

    # out will in the end point to a downsampled lowpassed image
    # out = ifft(fft(img, axis=1)
    #    * phi[0, :].reshape(1, N_nolp), axis=1)[:, ::dsf]

    # TO DO: use real transforms
    out = np.zeros((img.shape[:-1] + (N_nolp_r, 1)))
    out.shape = out.shape[:-1]
    # out = np.zeros((img.shape[0],img.shape[1], N_nolp_r))
    out[:] = np.fft.rfft(img, axis=-1) * phi[0, :N_nolp_r].reshape(1, N_nolp_r)

    out = np.fft.irfft(out, axis=-1)[..., ::dsf].copy()
    out = np.fft.rfft(out, axis=-2) * phi[:N_nolp_r, 0].reshape(N_nolp_r, 1)
    out = np.fft.irfft(out, axis=-2)[..., ::dsf, :]
    out = 2 ** (J - 1) * np.real(out)

    return out


"""
def apply_lowpass(img, phi, J, spatial_coef):

    #img can be

    N_nolp_r = img.shape[1]//2 + 1  #assuming squared images
    subsampling_rate = img.shape[1] / spatial_coef

    # out will in the end point to a downsampled lowpassed image
    # assuming filter phi is in the Fourier domain and it is separable (for instance Gaussian, Gabor)

    out = np.zeros((img.shape[0],img.shape[1], N_nolp_r))
    out[:] = np.fft.rfft(img, axis=2) * phi[0, :N_nolp_r].reshape(1, N_nolp_r)

    out = np.fft.irfft(out, axis=2)[:,:, ::subsampling_rate].copy()
    out = np.fft.rfft(out, axis=1) * phi[:N_nolp_r, 0].reshape(N_nolp_r, 1)
    out = np.fft.irfft(out, axis=1)[:,::subsampling_rate, :]
    out = 2**(J - 1) * np.real(out)

    return out
"""

def subsample(X,j):
    #assume X is in the spatial domain
    
    N = X.shape[3]
    d = np.arange(0,N,2.**(j-1))
    return X[:,:,:,d.astype(int),d.astype(int)]


def scattering(x,wavelet_filters):

    num_signals = x.shape[0]
    J = len(wavelet_filters['psi']) #number of scales
    L = len(wavelet_filters['psi'][0]) #number of orientations
    num_coefs = 1+J*L+J*(J-1)*L**2/2
    spatial_coefs = x.shape[1]/2**(J-1)

    U = []
    S = np.ndarray((num_signals,num_coefs,spatial_coefs,spatial_coefs))

    #Zero order coeffs
    S[:,0,:,:]= apply_lowpass(x, wavelet_filters['phi'][0], J,  spatial_coefs)

    #First order scattering coeffs
    Sview = S[:,1:J*L+1,:,:].view()
    Sview.shape=(num_signals,J,L,spatial_coefs,spatial_coefs)

    X = np.fft.fft2(x) # precompute the fourier transform of the images
    for j in np.arange(J):
        filtersj = wavelet_filters['psi'][j].view()
        # fft2(| x conv Psi_j |)
        U.append( np.abs(np.fft.ifft2(apply_fourier_mult(X,filtersj))))
        Sview[:, j, :, :, :] = apply_lowpass(U[j], wavelet_filters['phi'][0], J,  spatial_coefs)

    # Second order scattering coeffs
    sec_order_coefs = J*(J-1)*L**2/2

    S2norder = S[:,J*L+1:num_coefs,:,:] # view of the data
    S2norder.shape = (num_signals, sec_order_coefs/L, L, spatial_coefs, spatial_coefs)

    indx = 0
    for j1 in np.arange(J):
        Uj1 = np.fft.fft2(U[j1].view()) #U is in the spatial domain!
        for l1 in np.arange(Uj1.shape[1]):
            Ujl1 = Uj1[:,l1,].view()

            for j2 in np.arange(j1+1,J):
                # | U_lambda1 conv Psi_lambda2 | conv phi
                aux = np.abs(np.fft.ifft2(apply_fourier_mult(Ujl1, wavelet_filters['psi'][j2])))
                S2norder[:,indx,:,:,:]= \
                    apply_lowpass(aux, wavelet_filters['phi'][0], J,  spatial_coefs)

                indx = indx+1

    return S,U

