# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:33:09 2015

@author: bxiao
"""
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np

# read in an image using mis. 
img = misc.imread('cheetah.png',flatten=1)

f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# compute magnitude
magnitude_cheetah = 30*np.log(np.abs(fshift))
# compute phase
phase_cheetah = np.angle(fshift)

# read in an image using mis. 
img2 = misc.imread('zebra.png',flatten=1)
f = np.fft.fft2(img2)
fshift = np.fft.fftshift(f)
# compute magnitude
magnitude_zebra = 30*np.log(np.abs(fshift))
# compute phase
phase_zebra = np.angle(fshift)

# display the original image and the DFT components
plt.subplot(131),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_cheetah, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(phase_cheetah, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(131),plt.imshow(img2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(magnitude_zebra, cmap = 'gray')
plt.title('Log Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(phase_zebra, cmap = 'gray')
plt.title('Phase'), plt.xticks([]), plt.yticks([])
plt.show()

# reconstrution, please switch phase of zebra and cheeta, and display the final image here


