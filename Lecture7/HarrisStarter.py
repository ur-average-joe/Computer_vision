# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 12:53:09 2015

@author: bxiao
"""
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt

# write a 2D gaussian kernal
def matlab_style_gauss2D(shape=(3,3),sigma=1):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h =np.exp(-(x*x + y*y)/(2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
    
# construct derivative kernals from scratch
def gauss_derivative_kernels(size, sizey=None):
    
    return gx,gy    
    
# computing derivatives     
def gauss_derivatives(im, n, ny=None):
    """ returns x and y derivatives of an image using gaussian 
        derivative filters of size n. The optional argument 
        ny allows for a different size in the y direction."""

    gx,gy = gauss_derivative_kernels(n, sizey=ny)

    imx = signal.convolve(im,gx, mode='same')
    imy = signal.convolve(im,gy, mode='same')

    return imx,imy
    
gx,gy = gauss_derivative_kernels(3)


def compute_harris_response(image):
    """ compute the Harris corner detector response function 
        for each pixel in the image"""

    #derivatives
    imx,imy = gauss_derivatives(image, 3)

    #kernel for blurring
    gauss = matlab_style_gauss2D((3,3),3)

    #compute components of the structure tensor
 
    #determinant and trace
   

    return Wdet / Wtr

def get_harris_points(harrisim,min_dist=0, threshold=0.3):
    # find top corner candiates above a threshold

    
    #get the coordinates, all the non-zero components 

    
    # ...add their values

    
    # sort candidates in descending order of corner responses

    
    # store allowed point locations in array

    
    # select the best points taking min_dist into account
    
    
    return filtered_coords

# plotting the corners onto the image
def plot_harris_points(image,filtered_coords):
    plt.figure()
    plt.imshow(image,cmap='gray')
    plt.plot([p[1]for p in filtered_coords], [p[0] for p in filtered_coords],'*')
    plt.show()


# calling the harris corner detector
im = misc.imread('HarisCornerBuildling.png',flatten=1)
harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(harrisim,12,0.2)
plot_harris_points(im,filtered_coords)


