import numpy as np
import time
import sys
import mkl_fft

from utils import rotation_matrix_2d

def ispow2(N):

    return 0 == (N & (N - 1))


def periodize_filter(filt):

    if filt.dtype is np.dtype('float32'):
        cast = np.float32
    elif filt.dtype is np.dtype('complex64'):
        cast = np.complex64

    N = filt.shape[0]  # filter is square

    assert ispow2(N), 'Filter size must be an integer power of 2.'

    J = int(np.log2(N))
    filt_multires = dict()
    for j in range(J):
        # NTM: 0.5 is a cute trick for higher dimensions!
        mask = np.hstack((np.ones(N / 2**(1 + j)), 0.5, np.zeros(N - N / 2**(j + 1) - 1))) \
            + \
            np.hstack(
                (np.zeros(N - N / 2**(j + 1)), 0.5, np.ones(N / 2**(1 + j) - 1)))

        mask.shape = N, 1

        filt_lp = filt * mask * mask.T
        if 'cast' in locals():
            filt_lp = cast(filt_lp)

        # Remember: C contiguous, last index varies "fastest" (contiguous in
        # memory) (unlike Matlab)
        fold_size = (2**j, N / 2**j, 2**j, N / 2**j)
        filt_multires[j] = filt_lp.reshape(fold_size).sum(axis=(0, 2))

    return filt_multires


def fourier_multires(N, J=4, L=8, l1renorm=True, spiral=False, dtype='single', fft_choice='mkl_fft'):

    fft, ifft, fft2, ifft2, rfft, irfft = select_fft(fft_choice)

    assert ispow2(N), 'Filter size must be an integer power of 2.'

    lambda_list = [(j, l) for j in range(1, J + 1) for l in range(L)]
    filters = morlet_filter_bank_2d(
        (N, N), J=J, L=L, spiral=spiral, fft_choice=fft_choice)

    # Compute the lowpass filter phi at all resolutions
    phi_allres = periodize_filter(filters['phi'])

    filters_multires = dict(N=N, J=J, L=L,
                            resolution=range(J),
                            phi=phi_allres,
                            psi=dict(filt_list=[]))

    # Allocate filter memory for all resolutions
    for res in range(J):
        filters_multires['psi']['filt_list'].append(
            np.zeros((J * L,) + (N / 2**res, N / 2**res), dtype=dtype))

        # Address the filters in a nice way
        for i_lam, lam in enumerate(lambda_list):
            filters_multires['psi'][(lam, res)] = \
                filters_multires['psi']['filt_list'][res][i_lam]

    for lam in lambda_list:
        psi_allres = periodize_filter(filters['psi'][lam])

        # Copy the filter where it belongs at all resolutions
        for res in range(J):
            filters_multires['psi'][(lam, res)][:] = psi_allres[res]

            if l1renorm is True:
                filters_multires['psi'][(lam, res)][:] /= \
                    np.sum(np.abs(ifft2(filters_multires['psi'][(lam, res)])))

    return filters_multires


def select_fft(fft_choice):
    # TO DO: don't have this in various files. Have only one copy.

    if fft_choice == 'fftw':
        fft_module = pyfftw.interfaces.numpy_fft
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


from utils import rotation_matrix_2d


def gabor_2d(shape, sigma0, xi, theta, slant=None, pyramid=False, offset=None):
    """Returns a Gabor filter following the specifications

    This function creates a 2D complex Gabor filter. All parameters
    are taken to be in integer grid space


    Parameters
    ----------
    shape : {tuple, list}
        shape=(2,)
        Indicates the shape of the output array
    sigma_0 : float
        Indicates the standard deviation of the filter envelope along
        the oscillation axis
    slant : float
        Indicates the standard deviation of the filter envelope along
        the axis orthogonal to the oscillation. It is given relatively
        with respect to sigma_0 (sigma_orthog = slant * sigma_0)
    xi : float
        The oscillation wave number
    theta : float
        The oscillation wave orientation 
        (0 is a downward pointing wave vector)
    offset : {tuple, list, ndarray}
        shape=(2,)
        Possible offset for the filter from the origin. Defaults to 0


    See also
    --------
    Morlet wavelets

    """

    # to be compatible with the Matlab version
    shape = (shape[1], shape[0])

    if slant is None:
        slant = 1.
    if offset is None:
        if pyramid:
            offset = (shape[0]/2, shape[1]/2)
        else:
            offset = (-(-shape[0]/2), -(-shape[1]/2))

    offset = np.asanyarray(offset).reshape(2, 1, 1)

    # to understand the following, note that 5 / 2 = 2, -5 / 2 = -3
    g = np.mgrid[0:shape[0], 0:shape[1]]
    g -= offset
    print g[:3, :3, :]

    rot = rotation_matrix_2d(theta)
    invrot = np.linalg.inv(rot)
    rot_g = invrot.dot(g.reshape(2, -1))
    precision_matrix = np.diag([1., slant ** 2]) / (sigma0 ** 2)
    mahalanobis = (rot_g * (precision_matrix.dot(rot_g))).sum(0)

    raw_gabor = np.exp(-mahalanobis / 2 + 1j * xi * rot_g[0])

    if pyramid:
        gabor = raw_gabor.reshape(shape) / (2 * np.pi * sigma0**2 / slant**2)
    else:
        gabor = np.fft.fftshift(raw_gabor.reshape(shape)) / (
            2 * np.pi * sigma0**2 / slant**2)

    # return transposed in order to be compatible with matlab version
    return gabor.T


def morlet_2d_noDC(shape, sigma, xi, theta, slant=None, offset=None):

    gabor = gabor_2d(shape, sigma, xi, theta, slant, pyramid=False, offset=offset)
    envelope = np.abs(gabor)
    K = gabor.sum() / envelope.sum()

    centered = gabor - K * envelope

    return centered


def morlet_2d_pyramid(shape, sigma, xi, theta, slant=None, offset=None):

    gabor = gabor_2d(shape, sigma, xi, theta, slant, pyramid=True, offset=offset)
    envelope = np.abs(gabor)
    K = gabor.sum() / envelope.sum()

    centered = gabor - K * envelope

    return centered


def morlet_filter_bank_2d(shape, Q=1, L=8, J=4,
                          sigma_phi=.8,
                          sigma_psi=.8,
                          xi_psi=None,
                          slant_psi=None,
                          min_margin=None,
                          spiral=False,
                          dtype='single',
                          fft_choice='mkl_fft'):
    """Creates a multiscale bank of filters

    Creates and indexes filters at several scales and orientations

    Parameters
    ----------
    shape : {tuple, list, ndarray}
        shape=(2,)
        Tuple indicating the shape of the filters to be generated
    Q : int
        Number of scales per octave (constant-Q filter bank)
    J : int
        Total number of scales
    L : int
        Number of orientations
    sigma_phi : float
        standard deviation of low-pass mother wavelet
    sigma_psi : float
        standard deviation of the envelope of the high-pass psi_0
    xi_psi : float
        frequency peak of the band-pass mother wavelet
    slant_psi : float
        ratio between axes of elliptic envelope. Smaller means more
        orientation selective
    min_margin : int
        Padding for convolution

    """

    fft, ifft, fft2, ifft2, rfft, irfft = select_fft(fft_choice)

    # non-independent default values
    if xi_psi is None:
        xi_psi = .5 * np.pi * (2 ** (-1. / Q) + 1)
    if slant_psi is None:
        slant_psi = 4. / L
    if min_margin is None:
        min_margin = sigma_phi * 2 ** (float(J) / Q)

    # potentially do some padding here
    filter_shape = shape

    max_scale = 2 ** (float(J - 1) / Q)

    lowpass_spatial = np.real(gabor_2d(filter_shape, sigma_phi * max_scale,
                                       0., 0., 1.))
    lowpass_fourier = np.zeros(filter_shape, dtype=dtype)

    # TO DO: figure out why the following assignment doesn't work with mkl_fft
    lowpass_fourier = fft2(lowpass_spatial)

    little_wood_paley = np.zeros(lowpass_spatial.shape, dtype=dtype)

    lambda_list = [(j, l) for j in range(1, J + 1) for l in range(L)]

    filt_list = np.zeros((J * L,) + filter_shape, dtype=dtype)
    filters = dict(phi=lowpass_fourier, psi=dict(filt_list=filt_list),
                   lam=lambda_list, J=J, L=L, Q=Q)

    if spiral is False:
        angles = np.arange(L) * np.pi / L
    else:
        angles = np.arange(L) * 2 * np.pi / L

    for i_lam, lam in enumerate(lambda_list):
        if spiral is False:
            scale = 2 ** (float(lam[0] - 1) / Q)
        else:
            scale = 2 ** (float(lam[0] - 1) / Q + float(lam[1]) / L)

        angle = angles[lam[1]]

        band_pass_filter = filt_list[i_lam]  # this is a view
        filter_spatial = morlet_2d_noDC(filter_shape,
                                        sigma_psi * scale,
                                        xi_psi / scale,
                                        angle,
                                        slant_psi)

        band_pass_filter[:] = np.real(fft2(filter_spatial))
        filters['psi'][lam] = band_pass_filter

        # TO DO: be more careful here if not fourier_multires
        little_wood_paley += np.abs(band_pass_filter) ** 2

    little_wood_paley = np.fft.fftshift(little_wood_paley)
    lwp_max = little_wood_paley.max()

    for filt in filters['psi']['filt_list']:
        filt /= np.sqrt(lwp_max / 2)

    filters['littlewood_paley'] = little_wood_paley

    return filters


if __name__ == "__main__":
    q = morlet_2d_pyramid((16, 8), 0.9, 0.8, 1, None)    
    print np.abs(q[5:8, 5:8])
    print q.shape

    pass
