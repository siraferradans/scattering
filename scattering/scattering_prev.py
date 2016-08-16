import numpy as np
import skimage
from scipy import ndimage as ndi


def subsample(X,j):
    #assume X is in the spatial domain with first two
    #dims images, angle
    
    N = X.shape[3]
    dsf = 2**(j)
    
    return X[:,:,::dsf,::dsf]


def apply_fourier_mult(signals,filters):
# Assume signals and filters are in the Fourier domain
# and the dimensions are:
# - signals: (num_signals, Nj,Nj) (color images can be stacked in the num_signals)
# - filters: [L,N,N] where L is the num. of filters to apply

#We assume that the filter has a maximum frequency < Nj, so that we can crop it

# Goal pointwise multiplication

#    L = filters.shape[0]
    N = filters.shape[1]
    Nj= signals.shape[1]
    #    num_samples = signals.shape[0]

    filtered_signals=signals[:,np.newaxis,:,:] * filters[np.newaxis,:,:,:]



    """
        # NTM: 0.5 is a cute trick for higher dimensions!
        mask = np.hstack((np.ones(Nj), 0.5, np.zeros(N - Nj)))\
                     + \
                     np.hstack((np.zeros(N - Nj), 0.5, np.ones(Nj)))
        
        mask.shape = N, 1
              
              filt_lp = filt * mask * mask.T
              if 'cast' in locals():
              filt_lp = cast(filt_lp)
              
        """


    return filtered_signals


"""
    def crop_filter_for_subsampling(filt,j):
    
    cast = np.complex64
    N = filt.shape[0]  # filter is square
    
    assert ispow2(N), 'Filter size must be an integer power of 2.'
    
    J = int(np.log2(N))
    
    # NTM: 0.5 is a cute trick for higher dimensions!
    mask = np.hstack((np.ones(N / 2 ** (1 + j)), 0.5, np.zeros(N - N / 2 ** (j + 1) - 1))) \
    + \
    np.hstack(
    (np.zeros(N - N / 2 ** (j + 1)), 0.5, np.ones(N / 2 ** (1 + j) - 1)))
    
    mask.shape = N, 1
    
    filt_lp = filt * mask * mask.T
    if 'cast' in locals():
    filt_lp = cast(filt_lp)
    
    # Remember: C contiguous, last index varies "fastest" (contiguous in
    # memory) (unlike Matlab)
    fold_size = (2 ** j, N / 2 ** j, 2 ** j, N / 2 ** j)
    filt_multires = filt_lp.reshape(fold_size).sum(axis=(0, 2))
    
    
    return filt_multires
    """
"""
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
def apply_lowpass(img, phi, J, N_scat):

    Img_filtered = np.real(np.fft.ifft2(np.fft.fft2(img)*phi))
    ds = img.shape[-1]/2**J

    return 2 ** (J - 1) *Img_filtered[...,::ds,::ds].copy()


def scattering(x,wavelet_filters,m):
    
    num_signals = x.shape[0]
    J = len(wavelet_filters['psi'])    #number of scales
    L = len(wavelet_filters['psi'][0]) #number of orientations
    
    oversample = 1 #subsample at a rate a bit lower than the critic frequency
    
    
    if (m>1):
        num_coefs = 1+J*L+J*(J-1)*L**2/2
    else:
        num_coefs = 1+J*L


    spatial_coefs = x.shape[1] / 2 ** (J-1)

    U = []
    V = []
    v_resolution = []
    S = np.ndarray((num_signals,num_coefs,spatial_coefs,spatial_coefs))

    #Zero order coeffs
    S[:,0,:,:]= apply_lowpass(x, wavelet_filters['phi'][0], J,  spatial_coefs)

    #First order scattering coeffs
    Sview = S[:,1:J*L+1,:,:].view()
    Sview.shape=(num_signals,J,L,spatial_coefs,spatial_coefs)

    X = np.fft.fft2(x) # precompute the fourier transform of the images
    for j in np.arange(J):
        filtersj = wavelet_filters['psi'][j].view()
        resolution = max(j-oversample, 0)
        
        # fft2(| x conv Psi_j |): X is full resolution, as well as the filters
        #aux = subsample(np.fft.ifft2(apply_fourier_mult(X,filtersj)), resolution )
        aux = np.fft.ifft2(apply_fourier_mult(X,filtersj))
        
        V.append( aux )
        v_resolution.append(resolution)

        U.append( np.abs(aux))
        
        Sview[:, j, :, :, :] = apply_lowpass(U[j], wavelet_filters['phi'][0], J,  spatial_coefs)



    if (m>1):
        # Second order scattering coeffs
        sec_order_coefs = J*(J-1)*L**2/2
        
        S2norder = S[:,J*L+1:num_coefs,:,:] # view of the data
        S2norder.shape = (num_signals, sec_order_coefs/L, L, spatial_coefs, spatial_coefs)
        
        indx = 0
        for j1 in np.arange(J):
            Uj1 = np.fft.fft2(U[j1].view()) #U is in the spatial domain
            for l1 in np.arange(Uj1.shape[1]):
                Ujl1 = Uj1[:,l1,].view() #all images single angle, all spatial coefficients
                
                for j2 in np.arange(j1+1,J):
                    # | U_lambda1 conv Psi_lambda2 | conv phi
                    aux = np.abs(np.fft.ifft2(apply_fourier_mult(Ujl1, wavelet_filters['psi'][j2])))
                    #computing all angles at once
                    S2norder[:,indx,:,:,:]= \
                        apply_lowpass(aux, wavelet_filters['phi'][0], J,  spatial_coefs)
                    
                    indx = indx+1
    
#Rototranslation should use the V

    return S,U

"""
Scattering with no subsampling
def scattering(x,wavelet_filters,m):

    num_signals = x.shape[0]
    J = len(wavelet_filters['psi'])    #number of scales
    L = len(wavelet_filters['psi'][0]) #number of orientations
    
    if (m>1):
        num_coefs = 1+J*L+J*(J-1)*L**2/2
    else: 
        num_coefs = 1+J*L
        
        
    spatial_coefs = x.shape[1] / 2 ** (J-1)

    U = []
    V = []
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
        aux = subsample(np.fft.ifft2(apply_fourier_mult(X,filtersj)), j )
        V.append( aux )
        U.append( np.abs(aux))
        Sview[:, j, :, :, :] = apply_lowpass(U[j], wavelet_filters['phi'][0], J,  spatial_coefs)

    if (m>1):
        # Second order scattering coeffs
        sec_order_coefs = J*(J-1)*L**2/2

        S2norder = S[:,J*L+1:num_coefs,:,:] # view of the data
        S2norder.shape = (num_signals, sec_order_coefs/L, L, spatial_coefs, spatial_coefs)

        indx = 0
        for j1 in np.arange(J):
            Uj1 = np.fft.fft2(U[j1].view()) #U is in the spatial domain
            for l1 in np.arange(Uj1.shape[1]):
                Ujl1 = Uj1[:,l1,].view() #all images single angle, all spatial coefficients

                for j2 in np.arange(j1+1,J):
                    # | U_lambda1 conv Psi_lambda2 | conv phi
                    aux = np.abs(np.fft.ifft2(apply_fourier_mult(Ujl1, wavelet_filters['psi'][j2])))
                    #computing all angles at once
                    S2norder[:,indx,:,:,:]= \
                        apply_lowpass(aux, wavelet_filters['phi'][0], J,  spatial_coefs)

                    indx = indx+1

    #Rototranslation should use the V

    return S,U
"""
