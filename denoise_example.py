import numpy as np
from scipy.ndimage import filters
import rof

im = np.zeros((500, 500))
im[100:400,100:400] = 128
im[200:300,200:300] = 255
im = im + 30*np.random.standard_normal((500, 500))

U, T = rof.denoise(im, im)
G = filters.gaussian_filter(im, 10)

import pylab

pylab.figure()
pylab.gray()
pylab.imshow(im)
pylab.figure()
pylab.imshow(U)
pylab.show()
