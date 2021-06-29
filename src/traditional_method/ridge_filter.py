import numpy as np
import scipy


def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))
    
    freq_1d = np.reshape(freq, (1, rows*cols))
    ind = np.where(freq_1d > 0)
    
    ind = np.array(ind)
    ind = ind[1, :]
    
    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq*100)))/100
    
    unfreq = np.unique(non_zero_elems_in_freq)
    # print(ind)

    sigmax = 1/unfreq[0]*kx
    sigmay = 1/unfreq[0]*ky
    
    sze = int(np.round(3*np.max([sigmax, sigmay])))
    
    x, y = np.meshgrid(np.linspace(-sze, sze, (2*sze + 1)), np.linspace(-sze, sze, (2*sze + 1)))
    
    reffilter = np.exp(-(( (np.power(x,2))/(sigmax*sigmax) + (np.power(y,2))/(sigmay*sigmay)))) * np.cos(2*np.pi*unfreq[0]*x)
    
    filt_rows, filt_cols = reffilter.shape    
    
    gabor_filter = np.array(np.zeros((int(180 / angleInc), int(filt_rows), int(filt_cols))))
    
    for o in range(0, int(180/angleInc)):
                
        rot_filt = scipy.ndimage.rotate(reffilter, -(o*angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt
                    
    maxsze = int(sze)   

    temp = freq > 0
    validr, validc = np.where(temp)
    
    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze
    
    final_temp = temp1 & temp2 & temp3 & temp4    
    
    finalind = np.where(final_temp)
    
    maxorientindex = np.round(180/angleInc)
    orientindex = np.round(orient/np.pi*180/angleInc)
        
    for i in range(0, rows):
        for j in range(0, cols):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]
        
        img_block = im[r-sze:r+sze + 1][:, c-sze:c+sze + 1]
        
        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])
        
    return newim