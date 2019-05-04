import cv2
from PIL import Image
from pylab import *
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np 
from scipy.ndimage.filters import gaussian_filter 
from skimage import feature

'''Problem 1 (20pts). Warm up.  Download the image folder for PS1 and choose two images and do the following: for each of these images
'''
#1.1 Load the image into your environment
img = misc.imread("puppy.jpg",flatten=1)
img2 = misc.imread("einstein.png",flatten=1)



def problem1(img, img2):
    #1.2 Blur the image using Gaussian filter.
    img = ndimage.gaussian_filter(img, 2)

    img2 = ndimage.gaussian_filter(img2, 2)

    #1.3 Display the result.
    plt.imshow(img,cmap=plt.cm.gray)
    plt.show()

    plt.imshow(img2,cmap=plt.cm.gray)
    plt.show()

    #1.4 Compute the DFT (Discrete Fourier Transform) of the image. Please read Numpy FFT package. 
    # np.fft.fft2 and see here (http://docs.scipy.org/doc/numpy/reference/routines.fft.html)
    fft = np.fft.fft2(img)
    magnitude = 20*np.log(np.abs(fft))
    magnitude = np.asarray(magnitude, dtype=np.uint8)

    fft2 = np.fft.fft2(img2)
    magnitude2 = 20*np.log(np.abs(fft2))
    magnitude2 = np.asarray(magnitude2, dtype=np.uint8)

    #1.5 Display the magnitude of the DFT . 
    plt.plot(magnitude)
    plt.ylabel("magnitude")
    plt.show()

    plt.plot(magnitude2)
    plt.ylabel("magnitude2")
    plt.show()

#problem1 end
problem1(img, img2)


'''
Problem 2 (20 pts). Histogram equalization.  Compute the gray level (luminance) histogram for an image and equalize it so that the tones 
look better (and the image is less sensitive to exposure settings). You might want to use the following steps:

Convert the color image to luminance.
Compute the histogram, the cumulative distribution, and the compensation  transfer function (normalized CDF)

Try to increase the “punch” in the image by ensuring that a certain fraction of pixels (say 5%) are mapped to pure black and white. 
	Limit the local gain f’(I) in the transfer function. One way to do this is to limit 
f(I)<γI or f^'(I) <γ

while performing the accumulation, keeping any unaccumulated values “in reserve”.
'''

def problem2(img):
    histogram = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histogram)
    plt.show()

    cdf = histogram.cumsum()
    cdf_ = cdf * histogram.max()/ cdf.max()

    plt.plot(cdf_, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.show()


#problem2 end
problem2(img)


'''
Problem 3 (20pts) Separable filters.  Implement convolution with a separable kernel (Figure 3.14, Page 102 in http://szeliski.org/Book/).  
The input should be a grayscale or color image along with the horizontal and vertical kernels.  Please include examples of Gaussian filter, 
box filter, and Sobel filter. Make sure you support the padding mechanisms described in class and your textbook (chapter 3.2). You will need
 this functionality for some of the later exercise.  Please specify the kernels you used and display the original image and the output 
 images after each horizontal and vertical operation.   Here are a few padding options:

To compensate for this, a number of alternative padding or extension modes have been developed (Figure 3.13):
 • zero: set all pixels outside the source image to 0 (a good choice for alpha-matted cutout images);
 • constant (border color): set all pixels outside the source image to a specified border value; 
• clamp (replicate or clamp to edge): repeat edge pixels indefinitely; 
• (cyclic) wrap (repeat or tile): loop “around” the image in a “toroidal” configuration;

https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.ndimage.filters.convolve.html

You can use the optional parameter (mode)
'''
img = ndimage.gaussian_filter(img, 1)
def problem3_1(img):
    sobel_x =  ndimage.sobel(img, axis=0, mode="constant")
    sobel_y =  ndimage.sobel(img, axis=1, mode="constant") 

    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    plt.imshow(sobel_x)
    plt.show()

    plt.imshow(sobel_y)
    plt.show()

    plt.imshow(img)
    plt.show()

def problem3_2(img):
    convolve_x = np.array([0,1,-1])
    x = ndimage.convolve1d(img,convolve_x,axis= 0)
    sobel_convolve_x = ndimage.convolve(x,img)

    convolve_y = np.array([1,-1])
    y = ndimage.convolve1d(im,dx,axis= 1)
    sobel_convolve_y = ndimage.convolve(y,img)

    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

    plt.imshow(sobel_convolve_x)
    plt.show()

    plt.imshow(sobel_convolve_y)
    plt.show()

    plt.imshow(img)
    plt.show()


#problem3 is not finished""" Crashes at this point in the program. fix the bug """
problem3_1(img)
problem3_2(img)


'''
Problem 4 (10pts).  Write a function that finds the outline of simple objects in images (for example a square against white background) 
using image gradients. Your function input should be an image and output should be edges of the object.  Please write your own code with 
Scipy/Numpy. Do not use edge detection packages in OpenCV. 
'''
def problem4(img):
    edge_detection = feature.canny(img)
    edge_detection_2 = feature.canny(img, sigma=5)

    """ plt.imshow(img)
    plt.show() """

    plt.imshow(edge_detection,cmap=plt.cm.gray)
    plt.show()

    plt.imshow(edge_detection_2,cmap=plt.cm.gray)
    plt.show()

#problem4 end
problem4(img)

