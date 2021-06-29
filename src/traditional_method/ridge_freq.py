import numpy as np
from .frequest import frequest


def ridge_freq(im, mask, orient, blksze, windsze,minWaveLength, maxWaveLength):
    rows, cols = im.shape
    freq = np.zeros((rows, cols))
    
    for r in range(0, rows-blksze, blksze):
        for c in range(0, cols-blksze, blksze):
            blkim = im[r:r+blksze][:, c:c+blksze]
            blkor = orient[r:r+blksze][:, c:c+blksze]
            freq[r:r+blksze][:, c:c+blksze] = frequest(blkim, blkor, windsze, minWaveLength,
                                                       maxWaveLength)
    
    freq = freq*mask
    freq_1d = np.reshape(freq, (1, rows*cols))
    ind = np.where(freq_1d > 0)
    
    ind = np.array(ind)
    ind = ind[1, :]
    
    non_zero_elems_in_freq = freq_1d[0][ind]
    # print("non_zero_elems_in_freq = ", freq_1d)
    
    meanfreq = np.mean(non_zero_elems_in_freq)
    medianfreq = np.median(non_zero_elems_in_freq)
    return freq, meanfreq
