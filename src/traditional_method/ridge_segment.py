import numpy as np


def normalise(img, mean, std):
    normed = (img - np.mean(img))/(np.std(img))
    return normed


def ridge_segment(im, blksze, thresh):
    rows, cols = im.shape
    im = normalise(im, 0, 1)
    new_rows = np.int(blksze * np.ceil((np.float(rows))/(np.float(blksze))))
    new_cols = np.int(blksze * np.ceil((np.float(cols))/(np.float(blksze))))
    
    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))
    
    padded_img[0:rows][:, 0:cols] = im
    
    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i:i+blksze][:, j:j+blksze]
            
            stddevim[i:i+blksze][:, j:j+blksze] = np.std(block)*np.ones(block.shape)
    
    stddevim = stddevim[0:rows][:,0:cols]
                    
    mask = stddevim > thresh
    
    mean_val = np.mean(im[mask])
    
    std_val = np.std(im[mask])
    
    normim = (im - mean_val) / std_val
    
    return normim, mask
