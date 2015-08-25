from PIL import Image
import pca
import numpy as np
import pylab
import os

indir = 'data/a_thumbs'
imlist = [os.path.join(indir, f) for f in os.listdir(indir)]

im = np.array(Image.open(imlist[0]))
m, n = im.shape[0:2]
count = len(imlist)
immatrix = np.array([np.array(Image.open(i)).flatten() for i in imlist], 'f')

V, S, immean = pca.pca(immatrix)

pylab.figure()
pylab.gray()
pylab.subplot(2, 4, 1)
pylab.imshow(immean.reshape(m, n))
for i in range(7):
    pylab.subplot(2, 4, i+2)
    pylab.imshow(V[i].reshape(m, n))

pylab.show()
