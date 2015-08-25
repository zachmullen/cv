from scipy.ndimage import filters
import numpy as np
import pylab
from PIL import Image


def compute_harris_response(im, sigma=3):
    imy = np.zeros(im.shape)
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

    Wxx = filters.gaussian_filter(imx*imx, sigma)
    Wxy = filters.gaussian_filter(imx*imy, sigma)
    Wyy = filters.gaussian_filter(imy*imy, sigma)

    Wdet = Wxx*Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(im, min_dist=10, threshold=0.1):
    corner_threshold = im.max() * threshold
    harrisim_t = (im > corner_threshold) * 1

    coords = np.array(harrisim_t.nonzero()).T

    candidate_values = [im[c[0],c[1]] for c in coords]

    index = np.argsort(candidate_values)
    allowed_locations = np.zeros(im.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
                              (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0
    return filtered_coords


def get_descriptors(image, filtered_coords, wid=5):
    desc = []
    for coords in filtered_coords:
        desc.append(image[coords[0]-wid:coords[0]+wid+1,
                          coords[1]-wid:coords[1]+wid+1].flatten())
    return desc


def match(desc1, desc2, threshold=0.5):
    n = len(desc1[0])
    d = -np.ones((len(desc1),len(desc2)))

    for i, d1 in enumerate(desc1):
        for j, d2 in enumerate(desc2):
            d1 = (d1 - np.mean(d1)) / np.std(d1)
            d2 = (d2 - np.mean(d2)) / np.std(d2)
            ncc_value = np.sum(d1*d2) / (n-1)
            if ncc_value > threshold:
                d[i,j] = ncc_value

    return np.argsort(-d)[:,0]

def match_twosided(desc1, desc2, threshold=0.5):
    matches_12 = match(desc1, desc2, threshold)
    print 'matched one way'
    matches_21 = match(desc2, desc1, threshold)
    print 'matched two ways'

    ndx_12 = np.where(matches_12 >= 0)[0]

    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1

    return matches_12

def appendimages(im1, im2):
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1, im1.shape[1]))),axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2, im2.shape[1]))),axis=0)

    return np.concatenate((im1, im2), axis=1)

def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    im3 = appendimages(im1, im2)

    if show_below:
        im3 = np.vstack((im3, im3))

    pylab.imshow(im3)

    cols1 = im1.shape[1]
    for i,m in enumerate(matchscores):
        if m > 0:
            pylab.plot([locs1[i][1], locs2[m][1] + cols1],
                       [locs1[i][0],locs2[m][0]], 'c')
    pylab.axis('off')

if __name__ == '__main__':
    """im = np.array(Image.open('data/crans_1_small.jpg').convert('L'))
    harrisim = compute_harris_response(im)

    pts = get_harris_points(harrisim)

    pylab.figure()
    pylab.gray()
    pylab.imshow(im)

    pylab.plot([p[1] for p in pts], [p[0] for p in pts], 'r*')
    pylab.axis('off')
    pylab.show()"""

    im1 = np.array(Image.open('data/crans_1_small.jpg').convert('L'))
    im2 = np.array(Image.open('data/crans_2_small.jpg').convert('L'))
    wid = 5

    harrisim = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harrisim, wid+1)
    d1 = get_descriptors(im1, filtered_coords1, wid)

    harrisim = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harrisim, wid+1)
    d2 = get_descriptors(im2, filtered_coords2, wid)

    print('starting matching')
    matches = match_twosided(d1, d2)

    pylab.figure()
    pylab.gray()
    plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches[:100])
    pylab.show()
