# -*- coding: utf-8 -*-
"""
Simple & useful functions for EMPAD datasets.

Created on Sun Nov 24 13:15:33 2024

@author: Yu-Tsun Shao, USC Viterbi MFD. 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb
from scipy.ndimage import shift, fourier_shift
from skimage.filters import sobel, gaussian
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.registration import phase_cross_correlation
# from skimage.feature import register_translation
from scipy.ndimage import rotate
from scipy.io import loadmat
from scipy import fftpack
from tqdm import tqdm

#%%

def ReadRaw(fname):
    #Read .raw into numpy array
    with open(fname, 'rb') as file:
        dp = np.fromfile(file, np.float32)
    sqpix = dp.size/128/130  ##total slice
    pix = int(sqpix**(0.5))  ## scan steps (x,y) are equal

    dp = np.reshape(dp, (pix, pix, 130, 128), order = 'C')
    dp = dp[:, :, :128, :128]       ## crop out boundary dead pixels
    # dp = dp[:, :, 2:126, 2:126]
    where_are_NaNs = np.isnan(dp)   ## look for NAN and set as zero
    dp[where_are_NaNs] = 0
#    dp = dp[:,:,0:128,:]
    # print("dp shape: ", dp.shape)
    file.close()
    
    #Processes out low spurious counts and sets floor at just above zero to avoid errors when taking the logarithm
    #dp = dp*np.array(dp>20, np.int)
    
    dp[dp < 20] = 0
    dp = np.clip(dp, 1e-10, None) ##(min, max)
    return dp

def ReadRaw_Cropped(fname, shape):
    # shape = (yscan, xscan, ypix, xpix)
    yscan, xscan, ypix, xpix = shape    ## ypix = xpix = 128
    with open(fname, 'rb') as file:
        dp = np.fromfile(file, np.float32)
    dp = np.reshape(dp, (yscan, xscan, ypix, xpix), order = 'C')
    where_are_NaNs = np.isnan(dp)   ## look for NAN and set as zero
    dp[where_are_NaNs] = 0
    # print("dp shape: ", dp.shape)
    file.close()
    dp[dp < 20] = 0
    dp = np.clip(dp, 1e-10, None) ##(min, max)
    return dp    

def writeRaw(fname_out, dp):
    ## attempt to write into .raw for CSI tools
    dp.astype('float32').tofile(fname_out+".raw")
    print("done writing!")

def dp_avg(dp):
    yscan, xscan, ypix, xpix = dp.shape
    avg_dp = np.zeros((ypix,xpix))
    for i in tqdm(range(yscan)):
        for j in range(xscan):
            avg_dp += np.copy(dp[i,j,:,:])/(yscan*xscan)  ## np.copy() was necessary for computing dask arrays -- cannot modify the original array directly
    return avg_dp

def circular_mask(shape, centre, r0):
    ## follows the coord (y,x) in DM/ImageJ
    cy,cx = centre
    kx = np.arange(shape[-1])
    ky = np.arange(shape[-2])
    kx = kx -cx
    ky = ky -cy
    kx,ky = np.meshgrid(kx,ky)
    kdist = (kx**2.0 + ky**2.0)**(1.0/2)
    circ_mask = np.array(kdist <= r0, np.float32)
    return circ_mask

def ADF_mask(shape,centre, r1,r2):
    ## shape, eg. dp1.shape
    ## centre: (y_pix,x_pix)
    mask_c1 = circular_mask(shape, centre, r1)
    mask_c2 = circular_mask(shape, centre, r2)
    mask_ADF = mask_c2 - mask_c1
    return mask_ADF

def customDF(dp, mask):
    vdf = np.sum(mask*dp, axis = (-2,-1))
    return vdf

def rotateCropStack(dp, angle, flag_log):
    ## angle: >0: counter-clockwise
    ## rotate the stack counter-clockwise
    yscan, xscan, ysize, xsize = dp.shape
    dp2 = np.zeros((yscan, xscan, ysize, xsize))
    dp3 = dp_avg(dp)
    bkg = np.mean(np.sort(dp3[:,:].flatten())[0:10000])
    for i in tqdm(range(yscan)):
        for j in range(xscan):
            dp1 = rotate(dp[i,j,:,:], angle, reshape=False, mode='constant', cval=bkg)
#        dp2 = dp1[int(ypix/2-62):int(ypix/2+62), int(xpix/2-62):int(xpix/2+62)]
            if flag_log == 1:
                dp2[i,j,:,:] = np.log(dp1)
            else:
                dp2[i,j,:,:] = dp1
    dp2 = np.clip(dp2, 1e-10, None)
    return dp2