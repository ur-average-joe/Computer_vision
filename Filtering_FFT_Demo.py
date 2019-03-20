"""
Created on Wed Oct 3 19:33:09 2017

@author: bxiao
Here is a new comment
"""
import numpy as np
import pylab
from scipy import misc

# Take the 2-dimensional DFT and centre the frequencies
img = misc.imread('images/puppy.jpg',flatten=1)
# image size, square side length, number of squares
ncols, nrows = img.shape

# FFT the original image
ftimage = np.fft.fft2(img)
ftimage = np.fft.fftshift(ftimage)
pylab.imshow(np.log(np.abs(ftimage)),cmap='gray')
pylab.show()

# Build and apply a Gaussian filter and make sure the size is correct
sigmax, sigmay = 30, 30
cy, cx = nrows/2, ncols/2
x = np.linspace(0, nrows, nrows)
y = np.linspace(0, ncols, ncols)
X, Y = np.meshgrid(x, y)
pylab.imshow(X,cmap='gray')
pylab.show()

pylab.imshow(Y,cmap='gray')
pylab.show()
gmask = np.exp(-(((X-cx)/sigmax)**2 + ((Y-cy)/sigmay)**2))

pylab.imshow(gmask,cmap='gray')
pylab.show()


# here is filtering in Fourier Domain

ftimagep = ftimage * gmask

pylab.imshow(30*np.log(np.abs(ftimagep)),cmap='gray')
pylab.show()

# Finally, take the inverse transform and show the blurred image
imagep = np.fft.ifft2(ftimagep)
imageback = np.fft.ifftshift(imagep)
pylab.imshow(np.abs(imagep),cmap='gray')
pylab.show()


# Exercise, how do you highpass filter the image in FFT domain?
# step one: create a high pass filter in fourier domain
# Hint: many ways to create high pass filter 1) sobel, 2) subtract from low-pass
# 3) or something like this: 
#kernel = np.array([[-1, -1, -1],
 #                  [-1,  8, -1],
  #                 [-1, -1, -1]])






