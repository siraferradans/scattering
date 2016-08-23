import numpy as np
from skimage.filters import morlet_kernel, gabor_kernel


def _zero_pad_filter(filter, N):
    """ Zero pads 'filter' so it has a NxN size """

    if filter.shape[0] > N :
        M = np.array(filter.shape[0])
        init = np.int(np.floor(M / 2 - N / 2))
        filter = filter[init:init + N,:]

    if filter.shape[1] > N:
        M = np.array(filter.shape[1])
        init = np.int(np.floor(M / 2 - N / 2))
        filter = filter[:, init:init + N]

    left_pad = np.int64((np.array((N, N)) - np.array(filter.shape)) / 2)
    right_pad = np.int64(np.array((N, N)) - (left_pad + np.array(filter.shape)))

    padded_filter = np.lib.pad(filter, ((left_pad[0], right_pad[0]), (left_pad[1], right_pad[1])),
                               'constant', constant_values=(0, 0))
    return padded_filter


def multiresolution_filter_bank_morlet2d(N, J=4, L=8, sigma_phi = 0.8, sigma_xi = 0.8):

    wf, littlewood = filter_bank_morlet2d(N, J=J, L=L, sigma_phi=sigma_phi, sigma_xi=sigma_xi)

    multiresolution_wavelet_filters = filterbank_to_multiresolutionfilterbank(wf, J)

    return multiresolution_wavelet_filters, littlewood


def filter_bank_morlet2d(N, J=4, L=8, sigma_phi=0.8, sigma_xi=0.8):
    """ Compute a 2D complex Morlet filter bank [1]_ in the Fourier domain.

    Creates a filter bank of 1+JxL number of filters in the Fourier domain, where each filter has size NxN, and differ in
    the activation frequency. All these filters are complex 2D morlet filters.

    Parameters
    ----------
    N : size of the (squared) filters
    J : total number of scales of the filters which are located in the frequency domain, as powers of 2.
    L : total number of angles for each scale
    sigma_phi : standard deviation needed as a parameter for the low-pass filter (Gaussian)
    sigma_xi  : standard deviation needed as a parameter for every band-pass filter (Morlet)


    Returns
    -------
    Filters : Dictionary structure with the filters saved in the Fourier domain organized in the following way
            - Filters['phi'] : Low pass filter (Gaussian) in a 2D vector of size NxN
            - Filters['psi'] : Band pass filter (Morlet) saved as 4D complex array of size [J,L,N,N]
              where 'J' indexes the scale, 'L; the angles and NxN is the size of a single filter.


    littlewood_paley : Sum
    # tests the 'quality of the filters' by checking the littlewood-paley sum

    # Output : dictionary with the filters (in the Fourier domain), and the littlewood-paley image

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Filter_bank

    More information on Wavelet Filter banks can be found:
    https://en.wikipedia.org/wiki/Discrete_wavelet_transform

    Examples
    --------
        >>>> J = 3, L=8, px = 32
        >>>> wavelet_filters, littlewood = filter_bank_morlet2d(px, J=J, L=L, sigma_phi=0.6957,sigma_xi=0.8506 )



"""
    max_scale = 2 ** (float(J - 1))

    sigma = sigma_phi * max_scale
    freq = 0.

    filter_phi = np.ndarray((N,N), dtype='complex')
    littlewood_paley = np.zeros((N, N), dtype='single')

    # Low pass
    filter_phi = np.fft.fft2(np.fft.fftshift(_zero_pad_filter(gabor_kernel(freq, theta=0., sigma_x=sigma, sigma_y=sigma),N)))

    ## Band pass filters:
    ## psi: Create band-pass filters
    # constant values for psi
    xi_psi = 3. /4 * np.pi
    slant = 4. / L

    filters_psi = []

    for j, scale in enumerate(2. ** np.arange(J)):
        angles = np.zeros((L, N, N), dtype='complex')
        for l, theta in enumerate(np.pi * np.arange(L) / float(L)):
            sigma = sigma_xi * scale
            xi = xi_psi / scale

            sigma_x = sigma
            sigma_y = sigma / slant
            freq = xi / (np.pi * 2)

            psi = morlet_kernel(freq, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y,n_stds=12)
      
            #needs a small shift for odd sizes
            if (psi.shape[0] % 2 > 0):
                if (psi.shape[1] % 2 > 0):
                    Psi = _zero_pad_filter(psi[:-1, :-1], N)
                else:
                    Psi = _zero_pad_filter(psi[:-1, :], N)
            else:
                if (psi.shape[1] % 2 > 0):
                    Psi = _zero_pad_filter(psi[:, :-1], N)
                else:
                    Psi = _zero_pad_filter(psi, N)

            angles[l, :, :] = np.fft.fft2(np.fft.fftshift(0.5*Psi))

        littlewood_paley += np.sum(np.abs(angles) ** 2, axis=0)
        filters_psi.append(angles)

    lwp_max = littlewood_paley.max()

    for filt in filters_psi:
        filt /= np.sqrt(lwp_max/2)

    Filters = dict(phi=filter_phi, psi=filters_psi)

    return Filters, littlewood_paley*2



def filterbank_to_multiresolutionfilterbank(filters,Resolution):

    J = len(filters['psi']) #scales
    L = len(filters['psi'][0]) #angles
    N = filters['psi'][0].shape[-1] #size at max scale

    Phi_multires = []
    Psi_multires = []
    for res in np.arange(0,Resolution):
        Phi_multires.append(_get_filter_at_resolution(filters['phi'],res))

        aux_filt_psi = np.ndarray((J,L,int(N/2**res),int(N/2**res)), dtype='complex64')
        for j in np.arange(0,J):
            for l in np.arange(0,L):
                aux_filt_psi[j,l,:,:] = _get_filter_at_resolution(filters['psi'][j][l,:,:],res)

        Psi_multires.append(aux_filt_psi)


    Filters_multires = dict(phi=Phi_multires, psi=Psi_multires)
    return Filters_multires



def _ispow2(N):
    return 0 == (N & (N - 1))


def _get_filter_at_resolution(filt,j):

    cast = np.complex64
    N = filt.shape[0]  # filter is square

    assert _ispow2(N), 'Filter size must be an integer power of 2.'

    J = int(np.log2(N))

    # NTM: 0.5 is a cute trick for higher dimensions!
    mask = np.hstack((np.ones(int(N / 2 ** (1 + j))), 0.5, np.zeros(int(N - N / 2 ** (j + 1) - 1)))) \
           + \
           np.hstack(
               (np.zeros(int(N - N / 2 ** (j + 1))), 0.5, np.ones(int(N / 2 ** (1 + j) - 1))))

    mask.shape = N, 1

    filt_lp = filt * mask * mask.T
    if 'cast' in locals():
        filt_lp = cast(filt_lp)

    # Remember: C contiguous, last index varies "fastest" (contiguous in
    # memory) (unlike Matlab)
    fold_size = (int(2 ** j), int(N / 2 ** j), int(2 ** j), int(N / 2 ** j))
    filt_multires = filt_lp.reshape(fold_size).sum(axis=(0, 2))


    return filt_multires





