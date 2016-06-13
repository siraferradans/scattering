import scatterpyllar.filters.morlet_filter_bank_2d as spf
import scatterpyllar.filters.morlet_2d_noDC as morlet_2d_noDC
import scatterpyllar.core.scattering_simple as sp
import numpy as np
from scipy.misc import lena
import time
import cProfile

import sys
sys.path.append('/Users/doksa/Dropbox/Projects/mkl_fft')
import mkl_fft



# lena = lena() / 256.
# lena /= lena.max()

N = 256
J = int(np.log2(N))
L = 12


t = np.arange(N).reshape(N, 1)

x = np.float32(np.cos(t) * np.cos(t.T))
x[:,:] = 1

start_time = time.time()
# why doesn't mkl_fft work here?
fb = spf.fourier_multires(N, J=J, L=L, fft_choice='fftpack_lite')
print("--- %s seconds ---" % (time.time() - start_time))

# cProfile.run("sp.scattering_transform(x, fb, fft_choice='mklfft')")
n_mc = 1
start_time = time.time()

for i in range(n_mc):
    S, scr = sp.scattering_transform(x, fb, fft_choice='mkl_fft', localized=False)
print("--- %s seconds ---" % ((time.time() - start_time) / n_mc))


print S['coeffs'][((0, 0),(3, 1),(4, 3))]