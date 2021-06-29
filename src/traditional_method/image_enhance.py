from .ridge_segment import ridge_segment
from .ridge_orient import ridge_orient
from .ridge_freq import ridge_freq
from .ridge_filter import ridge_filter


def image_enhance(img):
    blksze = 16
    thresh = 0.1
    normim, mask = ridge_segment(img, blksze, thresh)


    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)


    blksze = 38
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength, maxWaveLength)
    
    
    freq = medfreq * mask
    # print(medfreq)
    kx = 0.65
    ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)
    
    return newim < -3