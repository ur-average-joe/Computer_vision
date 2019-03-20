# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:15:29 2015

@author: bxiao
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

# read in an image using mis. 
img = misc.imread('einstein.png',flatten=1)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 30*np.log(np.abs(fshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# removing the high frequencies 
rows, cols = img.shape
crow, ccol = rows/2 , cols/2     # center
# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1
plt.imshow(mask,cmap='gray')
plt.show()
print(mask.shape)

# apply the mask: this is the same as convolution in spacial domain!! 
##print fshiftnew.max()
# removing low frequency
fshift[crow-5:crow+5, ccol-5:ccol+5] = 0
print(fshift.max(),fshift.min())
magnitude_spectrum2 = 30*np.log(np.abs(fshift))
f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

# you need to image histogram your image here!



#this image needs to be normalized. 



plt.subplot(131),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_spectrum2 , cmap = 'gray')
plt.title('new Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(img_back, cmap='gray')
plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])
plt.show()                

## Exercise 1: how do you remove the high frequency?  Display the DFT magnitutde and the resulting image here


