from PIL import Image
import os
import numpy as np
import pylab
import subprocess
import harris

exe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'vlfeat', 'bin', 'maci64', 'sift')


def process_image(imagename, resultname, edge_thresh=10, peak_thresh=5):
    if imagename[-4:] != '.pgm':
        im = Image.open(imagename).convert('L')
        imagename = 'output/tmp.pgm'
        im.save(imagename)

    cmd = [str(arg) for arg in (
        exe, imagename, '--output', resultname, '--edge-thresh', edge_thresh,
        '--peak-thresh', peak_thresh
    )]

    subprocess.call(cmd)

    print('Created ' + resultname)

    return np.array(im)


def read_features_from_file(filename):
    f = np.loadtxt(filename)
    return f[:,:4], f[:,4:]


def plot_features(im, locs, circle=False):
    def draw_circle(c, r):
        t = np.arange(0,1.01,0.01)*2*np.pi
        x = r*np.cos(t) + c[0]
        y = r*np.sin(t) + c[1]
        pylab.plot(x,y,'b',linewidth=2)

    pylab.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2],p[2])
    else:
        pylab.plot(locs[:, 0], locs[:, 1], 'ob')
    pylab.axis('off')


def match_features(desc1, desc2):
    desc1 = np.array([d/np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d/np.linalg.norm(d) for d in desc2])

    d1_length = desc1.shape[0]

    matchscores = np.zeros((d1_length, 1), 'int')
    desc2t = desc2.T

    for i in range(d1_length):
        dotprods = 0.9999 * np.dot(desc1[i,:], desc2t)
        idx = np.argsort(np.arccos(dotprods))

        if np.arccos(dotprods)[idx[0]] < 0.6 * np.arccos(dotprods)[idx[1]]:
            matchscores[i] = int(idx[0])

    return matchscores


def match_twosided(desc1, desc2):
    print 'matching one way'
    matches_12 = match_features(desc1, desc2)
    print 'matching other way'
    matches_21 = match_features(desc2, desc1)

    print 'culling matches'
    idx_12 = matches_12.nonzero()[0]

    for i in idx_12:
        if matches_21[int(matches_12[i])] != i:
            matches_12[i] = 0

    return matches_12

if __name__ == '__main__':
    im1, im2 = [
        process_image('data/crans_%d_small.jpg' % i, 'output/im%d.sift' % i)
        for i in (1, 2)
    ]
    l1, d1 = read_features_from_file('output/im1.sift')
    l2, d2 = read_features_from_file('output/im2.sift')

    matches = match_features(d1, d2)
    print matches[:50]

    pylab.figure()
    pylab.gray()
    harris.plot_matches(im1, im2, l1[:,:2], l2[:,:2], matches)
    pylab.show()
