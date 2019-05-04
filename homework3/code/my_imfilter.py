# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:39:14 2015

@author: bxiao
"""
from __future__ import division
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy import misc
import math
import scipy
from scipy import*
from scipy.fftpack import*
from scipy.fftpack import fft, ifft, fft2



def convolution(A,B):
   #convolution of image and Kernel
	image_height =A.shape[0]
	image_width = A.shape[1]

	kernal_height = B.shape[0]
	kernal_width = B.shape[1]

	h = kernal_height/2
	w = kernal_width/2

   #numpy.dot 

	image_convolution = np.zeros(A.shape)
	for i in range(h,image_height - h):
		for j in range(w,image_width - w):#input 
			sum = 0
			for m in range(kernal_height):#kernel 
				for n in range(kernal_width):
					sum = sum + B[m][n] * A[i-h+m][j - w + n]
               
			sum = image_convolution[i][j]

	return image_convolution

def crosscorrelation(A,B):
   #return convolution(np.add(A,B))
	return convolution(np.conj(A),B[::-1])

""" def scaleSpectrum(A):
   return np.real(np.log10(np.absolute(A) + np.ones(A.shape)))
 """

# sample values from a spherical gaussian function from the center of the image
def makeGaussianFilter(numRows, numCols, sigma, highPass=True):
   centerI = int(numRows/2) + 1 if numRows % 2 == 1 else int(numRows/2)
   centerJ = int(numCols/2) + 1 if numCols % 2 == 1 else int(numCols/2)

   def gaussian(i,j):
      coefficient = math.exp(-1.0 * ((i - centerI)**2 + (j - centerJ)**2) / (2 * sigma**2))
      return 1 - coefficient if highPass else coefficient

   return np.array([[gaussian(i,j) for j in range(numCols)] for i in range(numRows)])


def filterDFT(imageMatrix, filterMatrix):
   shiftedDFT = fftshift(fft2(imageMatrix))
   misc.imsave("dft.png", scaleSpectrum(shiftedDFT))

   filteredDFT = shiftedDFT * filterMatrix
   misc.imsave("filtered-dft.png", scaleSpectrum(filteredDFT))
   return ifft2(ifftshift(filteredDFT))


def lowPass(imageMatrix, sigma):
   n,m = imageMatrix.shape

   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=False))


def highPass(imageMatrix, sigma):
   #subrtract  low pass from original
   #get high pas
   n,m = imageMatrix.shape
   return filterDFT(imageMatrix, makeGaussianFilter(n, m, sigma, highPass=True))


def hybridImage(highFreqImg, lowFreqImg, sigmaHigh, sigmaLow):
   highPassed = highPass(highFreqImg, sigmaHigh)
   lowPassed = lowPass(lowFreqImg, sigmaLow)
   #lowPassed = lowPassed[:,:-1]
   #A.shape(255,255)
   return highPassed + lowPassed #ValueError: operands could not be broadcast together with shapes (255,255) (194,259)


def playWithFiltering():
   marilyn = ndimage.imread("/Users/kevinfeyjoo/Desktop/Computer_vision/homework3/images/marilyn.png", flatten=True)

   highPassedMarilyn = highPass(marilyn, 20)
   lowPassedMarilyn = lowPass(marilyn, 20)

   misc.imsave("low-passed-marilyn.png", np.real(lowPassedMarilyn))
   misc.imsave("high-passed-marilyn.png", np.real(highPassedMarilyn))
   misc.imsave("sum-of-marilyns.png", np.real((highPassedMarilyn + lowPassedMarilyn)/2.0))
 
""" if __name__ == "__main__":
   einstein = ndimage.imread("/Users/kevinfeyjoo/Desktop/Computer_vision/homework3/images/einstein.png", flatten=True)
   marilyn = ndimage.imread("/Users/kevinfeyjoo/Desktop/Computer_vision/homework3/images/marilyn.png", flatten=True)

   hybrid = hybridImage(einstein, marilyn, 25, 10)
   misc.imsave("marilyn-einstein.png", np.real(hybrid)) """


einstein = ndimage.imread("/Users/kevinfeyjoo/Desktop/Computer_vision/homework3/images/einstein.png", flatten=True)
marilyn = ndimage.imread("/Users/kevinfeyjoo/Desktop/Computer_vision/homework3/images/marilyn.png", flatten=True)
img1 = convolution(einstein,marilyn)
img2 = crosscorrelation(einstein,marilyn)


highPass_einstein = highPass(einstein,50)
highPass_marilyn = highPass(marilyn,50)

lowPass_einstein = lowPass(einstein,10)
lowPass_marilyn = lowPass(marilyn,10)

hybridImage_based_e = hybridImage(einstein,marilyn,90,10)
hybridImage_based_h = hybridImage(marilyn,einstein,90,10)

plt.subplot(131),plt.imshow(marilyn, cmap = 'gray')
plt.title('DFT'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(shiftedDFT, cmap = 'gray')
plt.title('Filtered-DFT '), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(filteredDFT, cmap = 'gray')
plt.show()



plt.subplot(131),plt.imshow(hybridImage_based_h, cmap = 'gray')
plt.subplot(133),plt.imshow(hybridImage_based_e, cmap = 'gray')
plt.title('Hybrid Images'), plt.xticks([]), plt.yticks([])
plt.show()



#% This function is intended to behave like the built in function ndimage.filter. 
#% While terms like "filtering" and
#% "convolution" might be used interchangeably, and they are indeed nearly
#% the same thing, there is a difference:
#% from 'help filter2'
#%    2-D correlation is related to 2-D convolution by a 180 degree rotation
#%    of the filter matrix.
#
#% Your function should work for color images. Simply filter each color
#% channel independently.
#
#% Your function should work for filters of any width and height
#% combination, as long as the width and height are odd (e.g. 1, 7, 9). This
#% restriction makes it unambigious which pixel in the filter is the center
#% pixel.
#
#% Boundary handling can be tricky. The filter can't be centered on pixels
#% at the image boundary without parts of the filter being out of bounds. If
#% you look at 'help conv2' and 'help imfilter' you see that they have
#% several options to deal with boundaries. You should simply recreate the
#% default behavior of imfilter -- pad the input image with zeros, and
#% return a filtered image which matches the input resolution. A better
#% approach is to mirror the image content over the boundaries for padding.
#
#% % Uncomment if you want to simply call imfilter so you can see the desired
#% % behavior. When you write your actual solution, you can't use imfilter,
#% % filter2, conv2, etc. Simply loop over all the pixels and do the actual
#% % computation. It might be slow.
#% output = imfilter(image, filter);


