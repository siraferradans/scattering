import numpy as np
import skimage
from skimage.filters import morlet_kernel, gabor_kernel
from scipy import ndimage as ndi


def zero_pad_filter(filter, N):
    if (filter.shape[0] > N) :
        M = np.array(filter.shape[0])
        init = np.int(np.floor(M / 2 - N / 2))
        filter = filter[init:init + N,:]

    if (filter.shape[1] > N):
        M = np.array(filter.shape[1])
        init = np.int(np.floor(M / 2 - N / 2))
        filter = filter[:, init:init + N]

    left_pad = np.int64((np.array((N, N)) - np.array(filter.shape)) / 2)
    right_pad = np.int64(np.array((N, N)) - (left_pad + np.array(filter.shape)))

    padded_filter = np.lib.pad(filter, ((left_pad[0], right_pad[0]), (left_pad[1], right_pad[1])),
                               'constant', constant_values=(0, 0))
    return padded_filter


def filter_bank_morlet2d(N, J=4, L=8, sigma_phi = 0.8, sigma_xi = 0.8):
    # This function computes the set of morlet filters at the maximum size (NxN) and also
    # tests the 'quality of the filters' by checking the littlewood-paley sum

    # Output : dictionary with the filters (in the Fourier domain), and the littlewood-paley image


    # TODO:
    # - allow boundary values that are not circular
    # - introduce subsampling (now all filters are NxN)

    max_scale = 2 ** (float(J - 1))

    sigma = sigma_phi * max_scale
    freq = 0.

    filter_phi = np.ndarray((1,N,N), dtype='complex')
    littlewood_paley = np.zeros((N, N), dtype='single')

    # Low pass
    filter_phi[0, :, :] = np.fft.fft2(np.fft.fftshift(zero_pad_filter(gabor_kernel(freq, theta=0., sigma_x=sigma, sigma_y=sigma),N)))

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
                    Psi = zero_pad_filter(psi[:-1, :-1], N)
                else:
                    Psi = zero_pad_filter(psi[:-1, :], N)
            else:
                if (psi.shape[1] % 2 > 0):
                    Psi = zero_pad_filter(psi[:, :-1], N)
                else:
                    Psi = zero_pad_filter(psi, N)


            angles[l, :, :] = np.fft.fft2(np.fft.fftshift(Psi))

        littlewood_paley += np.sum(np.abs(angles) ** 2, axis=0)
        filters_psi.append(angles)


    lwp_max = littlewood_paley.max()

    for filt in filters_psi:
        filt /= np.sqrt(lwp_max/2)


    Filters = dict(phi=filter_phi, psi=filters_psi)

    return Filters, littlewood_paley



"""
def periodize_filter(filt,j):

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




