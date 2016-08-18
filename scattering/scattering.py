import numpy as np
import skimage
from scipy import ndimage as ndi


def apply_fourier_mult(signals,filters):
# Assume signals and filters are in the Fourier domain
# and the dimensions are:
# - signals: (num_signals, Nj,Nj) (color images can be stacked in the num_signals)
# - filters: [L,N,N] where L is the num. of filters to apply

    N = filters.shape[1]
    Nj= signals.shape[1]

    filtered_signals=signals[:,np.newaxis,:,:] * filters[np.newaxis,:,:,:]
    return filtered_signals


def subsample(X,j):
    #assume X is in the spatial domain with first two
    #dims images, angle
    
    N = X.shape[3]
    dsf = 2**(j)
    
    return 2**(j)*X[...,::dsf,::dsf]

def apply_lowpass(img, phi, J, N_scat):

    Img_filtered = np.real(np.fft.ifft2(np.fft.fft2(img)*phi))
#    ds = 2**(J-1) # -1 for the oversampling, to be sure that everything is ok
    N_nolp = img.shape[-1]

    N_nolp_r = N_nolp // 2 + 1
    ds = int(N_nolp / N_scat)
    
    return 2 ** (J - 1) *Img_filtered[...,::ds,::ds]

def scattering(x,wavelet_filters,m):
    
    num_signals = x.shape[0]
    J = len(wavelet_filters['psi'][0])    #number of scales
    L = len(wavelet_filters['psi'][0][0]) #number of orientations
    
    oversample = 1 #subsample at a rate a bit lower than the critic frequency
    
    if (m>1):
        num_coefs = int(1+J*L+J*(J-1)*L**2/2)
    else:
        num_coefs = int(1+J*L)


    spatial_coefs = int(x.shape[1] / 2 ** (J-1))

    U = []
    V = []
    v_resolution = []
    S = np.ndarray((num_signals,num_coefs,spatial_coefs,spatial_coefs))

    #print('J=',J,' L=',L,' num_signals=', num_signals, ' spatial_coefs=',spatial_coefs)

    current_resolution = 0

    #Zero order coeffs
    S[:,0,:,:] = apply_lowpass(x, wavelet_filters['phi'][current_resolution], J,  spatial_coefs)
    
    #First order scattering coeffs
    Sview = S[:,1:J*L+1,:,:].view()
    Sview.shape=(num_signals,J,L,spatial_coefs,spatial_coefs)

    X = np.fft.fft2(x) # precompute the fourier transform of the images
    for j in np.arange(J):
        filtersj = wavelet_filters['psi'][current_resolution][j].view()
        
        resolution = max(j-oversample, 0)# resolution for the next layer
        v_resolution.append(resolution) 

        # fft2(| x conv Psi_j |): X is full resolution, as well as the filters
        aux = subsample(np.fft.ifft2(apply_fourier_mult(X,filtersj)), resolution )
    
        #aux = np.fft.ifft2(apply_fourier_mult(X,filtersj))
        V.append( aux )
        U.append( np.abs(aux))

        #print('size Sview[',j,']=',Sview.shape)
        #print('size U[',j,']=',U[j].shape)
        Sview[:, j, :, :, :] = apply_lowpass(U[j], wavelet_filters['phi'][resolution], J,  spatial_coefs)

    # Second order scattering coeffs
    if (m>1):
        sec_order_coefs = int(J*(J-1)*L**2/2)
        
        S2norder = S[:,J*L+1:num_coefs,:,:] # view of the data
        S2norder.shape = (num_signals, int(sec_order_coefs/L), L, spatial_coefs, spatial_coefs)
        
        indx = 0
        for j1 in np.arange(J):
            Uj1 = np.fft.fft2(U[j1].view()) #U is in the spatial domain
            current_resolution = v_resolution[j1]
            
            for l1 in np.arange(Uj1.shape[1]):
                Ujl1 = Uj1[:,l1,].view() #all images single angle, all spatial coefficients
                
                for j2 in np.arange(j1+1,J):
                    # | U_lambda1 conv Psi_lambda2 | conv phi
                    aux = np.abs(np.fft.ifft2(apply_fourier_mult(Ujl1, wavelet_filters['psi'][current_resolution][j2])))
                    #computing all angles at once
                    S2norder[:,indx,:,:,:]= \
                        apply_lowpass(aux, wavelet_filters['phi'][current_resolution], J,  spatial_coefs)
                    
                    indx = indx+1
    
#Rototranslation should use the V

    return S,U

def apply_lowpass_fast(img, phi, J, N_scat):
    # NB: I could compute N_scat here, but in case we want to oversample, this
    # may be useful. I should make a class.

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


"""
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
